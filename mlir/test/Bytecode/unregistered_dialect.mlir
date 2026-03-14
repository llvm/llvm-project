// RUN: mlir-opt -emit-bytecode -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// verify that we round-trip an op without a dialect name (as long as we support this)
func.func @map1d(%lb: index, %ub: index, %step: index) {
// CHECK: "new_processor_id_and_range"
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)
  return
}
