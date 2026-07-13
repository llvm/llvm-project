// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s

// CHECK-LABEL: func.func @parallel_unstructured_loop
func.func @parallel_unstructured_loop(%buf: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32

  %dev = acc.copyin varPtr(%buf : memref<10xi32>) -> memref<10xi32>
  // CHECK-NOT: acc.loop
  // CHECK: acc.kernel_environment
  // CHECK-NOT: acc.par_width
  // CHECK: acc.compute_region
  // CHECK: scf.execute_region
  acc.parallel dataOperands(%dev : memref<10xi32>) {
    acc.loop {
    ^entry:
      cf.br ^header(%c0 : index)
    ^header(%iv: index):
      %cond = arith.cmpi ult, %iv, %c10 : index
      cf.cond_br %cond, ^body, ^exit
    ^body:
      memref.store %c1_i32, %dev[%iv] : memref<10xi32>
      %iv_next = arith.addi %iv, %c1 : index
      cf.br ^header(%iv_next : index)
    ^exit:
      acc.yield
    } attributes {independent = [#acc.device_type<none>], unstructured}
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<10xi32>) to varPtr(%buf : memref<10xi32>)
  return
}
