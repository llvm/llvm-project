; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -O0 -fast-isel=0 -global-isel=false %s -o - | FileCheck -enable-var-scope %s

define { i8, i1 } @test_cmpxchg_8(i8* %addr, i8 %desired, i8 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_8:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxrb [[OLD:w[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD]], w1, uxtb
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxrb [[STATUS:w[0-9]+]], w2, {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1, uxtb
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i8* %addr, i8 %desired, i8 %new seq_cst monotonic
  ret { i8, i1 } %res
}

define { i16, i1 } @test_cmpxchg_16(i16* %addr, i16 %desired, i16 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_16:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxrh [[OLD:w[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD]], w1, uxth
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxrh [[STATUS:w[3-9]]], w2, {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i16* %addr, i16 %desired, i16 %new seq_cst monotonic
  ret { i16, i1 } %res
}

define { i32, i1 } @test_cmpxchg_32(i32* %addr, i32 %desired, i32 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_32:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxr [[OLD:w[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD]], w1
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxr [[STATUS:w[0-9]+]], w2, {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  ret { i32, i1 } %res
}

define { i64, i1 } @test_cmpxchg_64(i64* %addr, i64 %desired, i64 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_64:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxr [[OLD:x[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD]], x1
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxr [[STATUS:w[0-9]+]], x2, {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{x[0-9]+}}, [[OLD]], x1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i64* %addr, i64 %desired, i64 %new seq_cst monotonic
  ret { i64, i1 } %res
}

define { i128, i1 } @test_cmpxchg_128(i128* %addr, i128 %desired, i128 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_128:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxp [[OLD_LO:x[0-9]+]], [[OLD_HI:x[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD_LO]], x2
; CHECK:     cset [[CMP_TMP:w[0-9]+]], ne
; CHECK:     cmp [[OLD_HI]], x3
; CHECK:     cinc [[CMP:w[0-9]+]], [[CMP_TMP]], ne
; CHECK:     cbnz [[CMP]], [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxp [[STATUS:w[0-9]+]], x4, x5, {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
  %res = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst monotonic
  ret { i128, i1 } %res
}

; Original implementation assumed the desired & new arguments had already been
; type-legalized into some kind of BUILD_PAIR operation and crashed when this
; was false.
@var128 = global i128 0
define {i128, i1} @test_cmpxchg_128_unsplit(i128* %addr) {
; CHECK-LABEL: test_cmpxchg_128_unsplit:
; CHECK:     mov [[ADDR:x[0-9]+]], x0
; CHECK:     add x[[VAR128:[0-9]+]], {{x[0-9]+}}, :lo12:var128
; CHECK:     ldp [[DESIRED_LO:x[0-9]+]], [[DESIRED_HI:x[0-9]+]], [x[[VAR128]]]
; CHECK:     ldp [[NEW_LO:x[0-9]+]], [[NEW_HI:x[0-9]+]], [x[[VAR128]]]
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxp [[OLD_LO:x[0-9]+]], [[OLD_HI:x[0-9]+]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cmp [[OLD_LO]], [[DESIRED_LO]]
; CHECK:     cset [[CMP_TMP:w[0-9]+]], ne
; CHECK:     cmp [[OLD_HI]], [[DESIRED_HI]]
; CHECK:     cinc [[CMP:w[0-9]+]], [[CMP_TMP]], ne
; CHECK:     cbnz [[CMP]], [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxp [[STATUS:w[0-9]+]], [[NEW_LO]], [[NEW_HI]], {{\[}}[[ADDR]]{{\]}}
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:

  %desired = load volatile i128, i128* @var128
  %new = load volatile i128, i128* @var128
  %val = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst seq_cst
  ret { i128, i1 } %val
}
