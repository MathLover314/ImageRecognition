using Images
using DataFrames
using CSV
using Colors
using DecisionTree
using ImageView

labelsInfo = DataFrame(CSV.File("trainLabels.csv"))


function imgConvert(image)
    resource = Float64.(Gray.(load(image)))
    temp = imresize(resource, (20, 20))
    if ndims(temp) == 3
        temp = mean(temp.data, 1)
    end
    reshape(temp, 1, 400)
end

x = zeros(size(labelsInfo, 1), 400)

for (index, idImage) in enumerate(labelsInfo[!,"ID"])
    nameFile = "train/train/$(idImage).Bmp"
    x[index, :] = imgConvert(nameFile)
end

y = map(c -> c[1], labelsInfo[!,"Class"])
y = Int.(y)

model = build_forest(y, x, 20, 50, 1.0)

test = zeros(6219, 400)
for i in 6284:12502
    nameFile = "test/test/$(i).Bmp"
    test[i-6283, :] = imgConvert(nameFile)
end

predTest = apply_forest(model, test)
predTest = Char.(predTest)
