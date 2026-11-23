// RUN: mlir-opt %s --convert-scf-to-cf \
// RUN:   --test-vector-warp-distribute=rewrite-warp-ops-to-scf-if | FileCheck %s

// Same data race as `@warp_op_inside_loop` in vector-warp-distribute.mlir,
// but with the loop lowered to a CFG cycle via `--convert-scf-to-cf`. The
// args buffer must still not be reused.

// CHECK-LABEL: func @warp_op_inside_cfg_loop(
// CHECK-SAME:     %[[lane:.*]]: index, %[[out:.*]]: memref<2x32xindex>)
func.func @warp_op_inside_cfg_loop(%lane: index, %out: memref<2x32xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %v = vector.broadcast %i : index to vector<2xindex>
    %r = gpu.warp_execute_on_lane_0(%lane)[32]
        args(%v : vector<2xindex>) -> (vector<1xindex>) {
      ^bb0(%arg: vector<64xindex>):
        %s = vector.extract_strided_slice %arg
            offsets = [0], sizes = [32], strides = [1]
            : vector<64xindex> to vector<32xindex>
        gpu.yield %s : vector<32xindex>
    }
    %x = vector.extract %r[0] : index from vector<1xindex>
    memref.store %x, %out[%i, %lane] : memref<2x32xindex>
  }
  return
}

// CHECK:   cf.cond_br
// CHECK:   %[[arg_buf:.*]] = memref.get_global @__shared_64xindex
// CHECK:   %[[res_buf:.*]] = memref.get_global @__shared_32xindex
// CHECK-NOT:   memref.subview
// CHECK:   scf.if
