func.func @test_extract_strided_slice_1(%arg0 : vector<4x4x4x4xf32>) -> vector<2x3x4x4xf32> {
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 3], strides = [1, 1], offsets = [1, 1]}
     : vector<4x4x4x4xf32> to vector<2x3x4x4xf32>
  return %0 : vector<2x3x4x4xf32>
}


func.func @test(%arg0 : vector<7xf32>, %arg1 : vector<3xf32>) -> vector<4xf32> {

  %out = vector.shuffle %arg0, %arg1 [0, 4, 5, 1]
    : vector<7xf32>, vector<3xf32> 
    return %out : vector<4xf32>

}
