// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

// `scf.parallel` conversion is not supported yet.
// Make sure that we do not accidentally invalidate this function by removing
// `scf.reduce`.
// CHECK-LABEL: func.func @func
// CHECK:         scf.parallel
// CHECK-NEXT:      spirv.Constant
// CHECK-NEXT:      memref.store
// CHECK-NEXT:      scf.reduce
// CHECK:         spirv.Return
func.func @func(%arg0: i64) {
  %0 = arith.index_cast %arg0 : i64 to index
  %alloc = memref.alloc() : memref<16xf32>
  scf.parallel (%arg1) = (%0) to (%0) step (%0) {
    %cst = arith.constant 1.000000e+00 : f32
    memref.store %cst, %alloc[%arg1] : memref<16xf32>
    scf.reduce
  }
  return
}
