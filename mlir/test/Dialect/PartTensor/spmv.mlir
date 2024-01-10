// RUN: mlir-opt %s -part-compiler
// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by mlir-opt (without any options.)

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#dense = #sparse_tensor.encoding<{
  lvlTypes = [ "dense"]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
#partDense = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #dense
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // X (in)
    affine_map<(i,j) -> (i,j)>,  // X (out)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func @spmv(%A: tensor<?x?xf32, #SortedCOO>,
                    %B: tensor<?xf32, #dense>,
                    %C: tensor<?xf32, #dense>) -> tensor<?xf32, #dense> {
    %c_m = arith.constant 0: index
    %c_k = arith.constant 1: index
    %c_n = arith.constant 1: index
    %sz_m = tensor.dim %A, %c_m:tensor<?x?xf32, #SortedCOO>
    %sz_n = tensor.dim %A, %c_n:tensor<?x?xf32, #SortedCOO>
    %D1 = bufferization.alloc_tensor(%sz_m) : tensor<?xf32, #dense>
    %D2 = linalg.matvec
      ins(%A, %B: tensor<?x?xf32, #SortedCOO>, tensor<?xf32, #dense>)
         outs(%D1: tensor<?xf32, #dense>) -> tensor<?xf32, #dense>
    %D = linalg.add
      ins(%D2, %C: tensor<?xf32, #dense>, tensor<?xf32, #dense>)
         outs(%D2: tensor<?xf32, #dense>) -> tensor<?xf32, #dense>
    return %D: tensor<?xf32, #dense>
  }
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>,
    %B: tensor<?xf32, #partDense>, %C: tensor<?xf32, #partDense>)
  {
    %a_partition_plan = part_tensor.get_partitions %A: tensor<?x?xf32, #partEncoding> -> memref<?xindex>
    %b_partition_plan = part_tensor.get_partitions %B: tensor<?xf32, #partDense> -> memref<?xindex>
    %c_partition_plan = part_tensor.get_partitions %C: tensor<?xf32, #partDense> -> memref<?xindex>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c2_index = arith.constant 2 : index
    %c3_index = arith.constant 3 : index
    %c4_index = arith.constant 4 : index
    %c9_index = arith.constant 9 : index
    %c2_i64 = arith.constant 2 : i64

    %A_num_points_index = part_tensor.get_num_partitions %A: tensor<?x?xf32, #partEncoding> -> index
    %B_num_points_index = part_tensor.get_num_partitions %B: tensor<?xf32, #partDense> -> index
    %C_num_points_index = part_tensor.get_num_partitions %C: tensor<?xf32, #partDense> -> index

    %C_Out_outer = scf.for %i = %c0 to %A_num_points_index step %c4_index
      iter_args(%O1 = %C) -> (tensor<?xf32, #partDense>) {

      %a_part_spec = memref.reinterpret_cast %a_partition_plan to offset: [%i], sizes: [%c4_index], strides: [%c1] : memref<?xindex> to memref<?xindex>
      %temp_spec = memref.alloc(%c4_index) : memref<?xindex>
      memref.copy %a_part_spec, %temp_spec: memref<?xindex> to memref<?xindex>
      %c_part_spec = memref.alloc(%c2_index) : memref<?xindex>
      %b_part_spec = memref.alloc(%c2_index) : memref<?xindex>
      %i_begin = memref.load %temp_spec[%c0_index] : memref<?xindex>
      %i_end = memref.load %temp_spec[%c2_index] : memref<?xindex>
      %j_begin = memref.load %temp_spec[%c1_index] : memref<?xindex>
      %j_end = memref.load %temp_spec[%c3_index] : memref<?xindex>
      memref.store %i_begin, %c_part_spec[%c0_index]: memref<?xindex>
      memref.store %i_end, %c_part_spec[%c1_index]: memref<?xindex>
      memref.store %j_begin, %b_part_spec[%c0_index]: memref<?xindex>
      memref.store %j_end, %b_part_spec[%c1_index]: memref<?xindex>

      %A_slice = part_tensor.get_slice %A, %a_part_spec : tensor<?x?xf32, #partEncoding>, memref<?xindex> -> tensor<?x?xf32, #SortedCOO>
      %B_slice = part_tensor.get_slice %B, %b_part_spec : tensor<?xf32, #partDense>, memref<?xindex> -> tensor<?xf32, #dense>
      %C_slice = part_tensor.get_slice %O1, %c_part_spec : tensor<?xf32, #partDense>, memref<?xindex> -> tensor<?xf32, #dense>

      %c_out = func.call @spmv(%A_slice, %B_slice, %C_slice):
        (tensor<?x?xf32, #SortedCOO>, tensor<?xf32, #dense>,
          tensor<?xf32, #dense>) -> tensor<?xf32, #dense>

      %O2 = part_tensor.update_slice %O1, %c_part_spec, %c_out , addf :
        tensor<?xf32, #partDense>, memref<?xindex>, tensor<?xf32, #dense>
          -> tensor<?xf32, #partDense>
      scf.yield %O2 :  tensor<?xf32, #partDense>
    }
    return
  }
}
