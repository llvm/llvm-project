; RUN: llc -mtriple=aarch64-- -O0 -fast-isel -fast-isel-abort=4 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: cmpxchg_monotonic_32:
; CHECK:          mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9_]+]]:
; CHECK-NEXT:     ldaxr w0, [[[ADDR]]]
; CHECK-NEXT:     cmp w0, w1
; CHECK-NEXT:     b.ne [[DONE:.LBB[0-9_]+]]
; CHECK-NEXT: // %bb.2:
; CHECK-NEXT:     stlxr [[STATUS:w[0-9]+]], w2, [[[ADDR]]]
; CHECK-NEXT:     cbnz [[STATUS]], [[RETRY]]
; CHECK-NEXT: [[DONE]]:
; CHECK-NEXT:     cmp w0, w1
; CHECK-NEXT:     cset [[STATUS]], eq
; CHECK-NEXT:     and [[STATUS32:w[0-9]+]], [[STATUS]], #0x1
; CHECK-NEXT:     str [[STATUS32]], [x3]
define i32 @cmpxchg_monotonic_32(i32* %p, i32 %cmp, i32 %new, i32* %ps) #0 {
  %tmp0 = cmpxchg i32* %p, i32 %cmp, i32 %new monotonic monotonic
  %tmp1 = extractvalue { i32, i1 } %tmp0, 0
  %tmp2 = extractvalue { i32, i1 } %tmp0, 1
  %tmp3 = zext i1 %tmp2 to i32
  store i32 %tmp3, i32* %ps
  ret i32 %tmp1
}

; CHECK-LABEL: cmpxchg_acq_rel_32_load:
; CHECK:      // %bb.0:
; CHECK:          mov [[ADDR:x[0-9]+]], x0
; CHECK:          ldr [[NEW:w[0-9]+]], [x2]
; CHECK-NEXT: [[RETRY:.LBB[0-9_]+]]:
; CHECK-NEXT:     ldaxr w0, [[[ADDR]]]
; CHECK-NEXT:     cmp w0, w1
; CHECK-NEXT:     b.ne [[DONE:.LBB[0-9_]+]]
; CHECK-NEXT: // %bb.2:
; CHECK-NEXT:     stlxr [[STATUS:w[0-9]+]], [[NEW]], [[[ADDR]]]
; CHECK-NEXT:     cbnz [[STATUS]], [[RETRY]]
; CHECK-NEXT: [[DONE]]:
; CHECK-NEXT:     cmp w0, w1
; CHECK-NEXT:     cset [[STATUS]], eq
; CHECK-NEXT:     and [[STATUS32:w[0-9]+]], [[STATUS]], #0x1
; CHECK-NEXT:     str [[STATUS32]], [x3]
define i32 @cmpxchg_acq_rel_32_load(i32* %p, i32 %cmp, i32* %pnew, i32* %ps) #0 {
  %new = load i32, i32* %pnew
  %tmp0 = cmpxchg i32* %p, i32 %cmp, i32 %new acq_rel acquire
  %tmp1 = extractvalue { i32, i1 } %tmp0, 0
  %tmp2 = extractvalue { i32, i1 } %tmp0, 1
  %tmp3 = zext i1 %tmp2 to i32
  store i32 %tmp3, i32* %ps
  ret i32 %tmp1
}

; CHECK-LABEL: cmpxchg_seq_cst_64:
; CHECK: mov [[ADDR:x[0-9]+]], x0
; CHECK: [[RETRY:.LBB[0-9_]+]]:
; CHECK-NEXT:     ldaxr x0, [[[ADDR]]]
; CHECK-NEXT:     cmp x0, x1
; CHECK-NEXT:     b.ne [[DONE:.LBB[0-9_]+]]
; CHECK-NEXT: // %bb.2:
; CHECK-NEXT:     stlxr [[STATUS]], x2, [[[ADDR]]]
; CHECK-NEXT:     cbnz [[STATUS]], [[RETRY]]
; CHECK-NEXT: [[DONE]]:
; CHECK-NEXT:     cmp x0, x1
; CHECK-NEXT:     cset [[STATUS:w[0-9]+]], eq
; CHECK-NEXT:     and [[STATUS32:w[0-9]+]], [[STATUS]], #0x1
; CHECK-NEXT:     str [[STATUS32]], [x3]
define i64 @cmpxchg_seq_cst_64(i64* %p, i64 %cmp, i64 %new, i32* %ps) #0 {
  %tmp0 = cmpxchg i64* %p, i64 %cmp, i64 %new seq_cst seq_cst
  %tmp1 = extractvalue { i64, i1 } %tmp0, 0
  %tmp2 = extractvalue { i64, i1 } %tmp0, 1
  %tmp3 = zext i1 %tmp2 to i32
  store i32 %tmp3, i32* %ps
  ret i64 %tmp1
}

attributes #0 = { nounwind }
