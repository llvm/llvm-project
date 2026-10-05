// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-xegpu="max-promoted-buffer-bytes=8" -split-input-file | FileCheck %s --check-prefix=CAP

// End-to-end check that convert-vector-to-xegpu promotes a whole-buffer
// accumulator carried across scf.for into a vector SSA value (threaded as an
// iter_arg/result) before lowering, so nothing is spilled to memory.

// CHECK-LABEL: func.func @acc_in_loop
//    CHECK-NOT:   memref.alloc
//    CHECK-NOT:   vector.transfer
//    CHECK-NOT:   xegpu.
//        CHECK:   %[[RES:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (vector<4xf32>)
//        CHECK:     %[[NEXT:.*]] = arith.addf %[[IT]], %[[IT]] : vector<4xf32>
//        CHECK:     scf.yield %[[NEXT]] : vector<4xf32>
//        CHECK:   return %[[RES]] : vector<4xf32>

// With an 8-byte cap the 16-byte buffer is not promoted and lowers to XeGPU
// memory ops instead.
// CAP-LABEL: func.func @acc_in_loop
//       CAP:   scf.for
//       CAP:   xegpu.{{load|store}}
func.func @acc_in_loop(%pad: f32, %lb: index, %ub: index, %step: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  scf.for %i = %lb to %ub step %step {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
    %n = arith.addf %v, %v : vector<4xf32>
    vector.transfer_write %n, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  }
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}
