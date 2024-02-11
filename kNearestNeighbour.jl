x = x'
test = test'

function distance(a, b)
    dot(a-b, a-b)
end

function neighbor(z, i, k)
    nrows, ncols = size(z)
    imageI = zeros(Float64, nrows)

    for index in 1:nrows
        imageI[index] = z[index, i]
    end

    imageJ = zeros(Float32, nrows)
    distances = zeros(Float32, ncols)

    for j in 1:ncols
        for index in 1:nrows
            imageJ[index] = z[index, j]
        end
    
        distances[j] = distance(imageI, imageJ)
    end
    sortedNeighbors = sortperm(distances)
    sortedNeighbors[2:k+1]
end

function assign_label(z, y, k, i) 
    knn = neighbor(z, i, k)
    counts = Dict{Int, Int}()

    highestCount = 0
    mostPopularLabel = 0

    for n in knn
        labelOfN = y[n]
        if !haskey(counts, labelOfN)
            counts[labelOfN] = 0
        end
        counts[labelOfN] += 1

        if counts[labelOfN] > highestCount
            highestCount = counts[labelOfN]
            mostPopularLabel = labelOfN
        end
    end
    mostPopularLabel
end

prediction = [assign_label(x, y, 1, i) for i in 1:size(x, 2)]

k = 1
using Distributed
sumValues = @distributed (+) for i = 1:size(x, 2) 
    assign_label(x, y, k, i) == y[i, 1]
end
loofCvAccuracy = sumValues / size(x, 2)

v = Char.(prediction)