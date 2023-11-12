// RUN: mlir-opt %s -part-compiler
// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by mlir-opt (without any options.)

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>) -> index {
    %num_points_index = part_tensor.get_num_partitions %A : tensor<?x?xf32, #partEncoding> -> index
    return %num_points_index: index
  }
}
