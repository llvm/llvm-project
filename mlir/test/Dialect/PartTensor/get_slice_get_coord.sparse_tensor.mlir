// RUN: mlir-opt %s --part-compiler=enable-runtime-library=true
// this is test case for goal for week of 11/9. This is how the lowered code
// should look like.

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
module {
  func.func @dumpPartitions(%A: !llvm.ptr) -> memref<?xindex> {
    %partition_plan = call @getPartitions(%A): (!llvm.ptr) -> memref<?xindex>
    %first_part_spec = memref.cast %partition_plan:
      memref<?xindex> to memref<4xindex>
    %slice = call @getSlice(%A, %first_part_spec):
      (!llvm.ptr, memref<4xindex>) -> tensor<?x?xf32, #SortedCOO>
    %res = sparse_tensor.coordinates %slice { level = 1 : index } :
      tensor<?x?xf32, #SortedCOO>  to memref<?xindex>
    return %res : memref<?xindex>
  }
  func.func private @getPartitions(!llvm.ptr) ->
    memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @getSlice(!llvm.ptr, memref<4xindex>) ->
    tensor<?x?xf32, #SortedCOO> attributes {llvm.emit_c_interface}
}
