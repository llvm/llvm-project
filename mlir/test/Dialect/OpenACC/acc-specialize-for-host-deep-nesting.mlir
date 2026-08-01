// RUN: mlir-opt %s -acc-specialize-for-host | FileCheck %s

// Deeply nested orphan acc.loops in a seq routine require more than the
// greedy rewriter's default iteration cap to fully lower to scf.for. The
// pass must run the rewrite to convergence rather than bailing out.

acc.routine @acc_routine_deep func(@deeply_nested_orphan_loops) seq
// CHECK-LABEL:   func.func @deeply_nested_orphan_loops
// CHECK-NOT:       acc.loop
// CHECK-COUNT-16: scf.for
func.func @deeply_nested_orphan_loops(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_deep]>} {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  acc.loop control(%iv0 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
    acc.loop control(%iv1 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
      acc.loop control(%iv2 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
        acc.loop control(%iv3 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
          acc.loop control(%iv4 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
            acc.loop control(%iv5 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
              acc.loop control(%iv6 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                acc.loop control(%iv7 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                  acc.loop control(%iv8 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                    acc.loop control(%iv9 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                      acc.loop control(%iv10 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                        acc.loop control(%iv11 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                          acc.loop control(%iv12 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                            acc.loop control(%iv13 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                              acc.loop control(%iv14 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                                acc.loop control(%iv15 : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
                                  memref.store %iv15, %arg0[] : memref<i32>
                                acc.yield
                                } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                              acc.yield
                              } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                            acc.yield
                            } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                          acc.yield
                          } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                        acc.yield
                        } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                      acc.yield
                      } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                    acc.yield
                    } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                  acc.yield
                  } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
                acc.yield
                } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
              acc.yield
              } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
            acc.yield
            } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
          acc.yield
          } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
        acc.yield
        } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
      acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
    acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
  acc.yield
  } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
  return
}
