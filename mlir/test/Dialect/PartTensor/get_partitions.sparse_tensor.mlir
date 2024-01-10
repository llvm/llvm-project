// RUN: mlir-opt --sparse-compiler=enable-runtime-library=true %s
// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by mlir-opt (without any options.)

module {
  func.func @dumpPartitions(%A: !llvm.ptr) -> memref<?xindex> {
    %partition_plan = call @getPartitions(%A): (!llvm.ptr) -> memref<?xindex>
    return %partition_plan: memref<?xindex>
  }
  func.func private @getPartitions(!llvm.ptr) -> memref<?xindex> attributes {llvm.emit_c_interface}
}
