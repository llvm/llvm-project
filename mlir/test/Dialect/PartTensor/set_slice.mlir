// RUN: mlir-opt %s -part-compiler

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module{
func.func @getPartitions(%A: tensor<?x?xf32, #partEncoding>, %B: tensor<?xindex>) -> index {
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %num_points_index = part_tensor.get_num_partitions %A : tensor<?x?xf32, #partEncoding> -> index
      part_tensor.set_slice %A, %A, %B : tensor<?x?xf32, #partEncoding>, tensor<?x?xf32, #partEncoding>, tensor<?xindex>
    return %num_points_index: index
}
}
