; RUN: llc -march=bpfel < %s | FileCheck %s
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
; CHECK-NOT: call __multi3
; CHECK: exit
