; RUN: llc -march=bpfel -mattr=+allow-builtin-calls < %s | FileCheck %s
;
; C code for this test case:
;
; long func(long a, long b) {
;     long x;
;     return __builtin_mul_overflow(a, b, &x);
; }


declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)

define noundef range(i64 0, 2) i64 @func(i64 noundef %a, i64 noundef %b) local_unnamed_addr {
entry:
  %0 = tail call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %1 = extractvalue { i64, i1 } %0, 1
  %conv = zext i1 %1 to i64
  ret i64 %conv
}

; CHECK-LABEL: func
; CHECK: r4 = r2
; CHECK: r2 = r1
; CHECK: r3 = r2
; CHECK: r3 s>>= 63
; CHECK: r5 = r4
; CHECK: r5 s>>= 63
; CHECK: r1 = r10
; CHECK: r1 += -16
; CHECK: call __multi3
; CHECK: r1 = *(u64 *)(r10 - 16)
; CHECK: r1 s>>= 63
; CHECK: w0 = 1
; CHECK: r2 = *(u64 *)(r10 - 8)
; CHECK: if r2 != r1 goto LBB0_2
; CHECK:  # %bb.1:                                # %entry
; CHECK: w0 = 0
; CHECK:  LBB0_2:                                 # %entry
; CHECK: exit