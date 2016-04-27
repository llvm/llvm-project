; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -O0 %s -o - | FileCheck %s

define { i8, i1 } @test_cmpxchg_8(i8* %addr, i8 %desired, i8 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_8:
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxrb [[OLD:w[0-9]+]], [x0]
; CHECK:     cmp [[OLD]], w1, uxtb
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxrb [[STATUS:w[3-9]]], w2, [x0]
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i8* %addr, i8 %desired, i8 %new seq_cst monotonic
  ret { i8, i1 } %res
}

define { i16, i1 } @test_cmpxchg_16(i16* %addr, i16 %desired, i16 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_16:
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxrh [[OLD:w[0-9]+]], [x0]
; CHECK:     cmp [[OLD]], w1, uxth
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxrh [[STATUS:w[3-9]]], w2, [x0]
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i16* %addr, i16 %desired, i16 %new seq_cst monotonic
  ret { i16, i1 } %res
}

define { i32, i1 } @test_cmpxchg_32(i32* %addr, i32 %desired, i32 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_32:
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxr [[OLD:w[0-9]+]], [x0]
; CHECK:     cmp [[OLD]], w1
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxr [[STATUS:w[3-9]]], w2, [x0]
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{w[0-9]+}}, [[OLD]], w1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  ret { i32, i1 } %res
}

define { i64, i1 } @test_cmpxchg_64(i64* %addr, i64 %desired, i64 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_64:
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxr [[OLD:x[0-9]+]], [x0]
; CHECK:     cmp [[OLD]], x1
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxr [[STATUS:w[3-9]]], x2, [x0]
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     subs {{x[0-9]+}}, [[OLD]], x1
; CHECK:     cset {{w[0-9]+}}, eq
  %res = cmpxchg i64* %addr, i64 %desired, i64 %new seq_cst monotonic
  ret { i64, i1 } %res
}

define { i128, i1 } @test_cmpxchg_128(i128* %addr, i128 %desired, i128 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_128:
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxp [[OLD_LO:x[0-9]+]], [[OLD_HI:x[0-9]+]], [x0]
; CHECK:     cmp [[OLD_LO]], x2
; CHECK:     sbcs xzr, [[OLD_HI]], x3
; CHECK:     b.ne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     stlxp [[STATUS:w[0-9]+]], x4, x5, [x0]
; CHECK:     cbnz [[STATUS]], [[RETRY]]
; CHECK: [[DONE]]:
  %res = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst monotonic
  ret { i128, i1 } %res
}
