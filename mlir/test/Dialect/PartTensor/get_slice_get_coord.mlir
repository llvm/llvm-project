// RUN: mlir-opt %s
// this is test case for goal for week of 11/9

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>)
    -> memref<?xindex>
  {
    %partition_plan = part_tensor.get_partitions %A:
      tensor<?x?xf32, #partEncoding> -> memref<?xindex>
    %first_part_spec = memref.cast %partition_plan:
      memref<?xindex> to memref<4xindex>
    %slice = part_tensor.get_slice %A, %first_part_spec:
      tensor<?x?xf32, #SortedCOO>
    %res = sparse_tensor.coordinates %slice { level = 1 : index } :
      tensor<?x?xf32, #SortedCOO>  to memref<?xindex>
    return %res : memref<?xindex>
  }
}
