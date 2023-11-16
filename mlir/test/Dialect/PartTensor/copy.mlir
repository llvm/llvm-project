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
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>) {
    %partition_plan = part_tensor.get_partitions %A:
	  tensor<?x?xf32, #partEncoding> -> memref<?xindex>
	%c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %c2_i64 = arith.constant 2 : i64
    %num_points_index = part_tensor.get_num_partitions %A:
      tensor<?x?xf32, #partEncoding> -> index
	scf.for %i = %c0 to %num_points_index step %c4_index{
      %part_spec = memref.reinterpret_cast %partition_plan to
        offset: [%i], sizes: [%c4_index], strides: [%c1]
        : memref<?xindex> to memref<?xindex>
	  %A_slice = part_tensor.get_slice %A, %part_spec :
        tensor<?x?xf32, #partEncoding>, memref<?xindex> ->
          tensor<?x?xf32, #SortedCOO>
	  %c0f = arith.constant 0.0 : f32
      %relu_out = linalg.generic #relu_memory_access_map
      outs(%A_slice: tensor<?x?xf32, #SortedCOO>) {
        ^bb(%x: f32):
          %1 = arith.maxf %x, %c0f : f32
          linalg.yield %1 : f32
    	} -> tensor<?x?xf32, #SortedCOO>
 	}
    return
  }
}
