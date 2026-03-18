; RUN: not llc -mtriple=bpf -mcpu=v3 < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function foo i64 (i32, i32, i32, i32, [2 x i64]): aggregate argument is split between registers and stack

; Source code:
;   struct t { long a; long b; };
;
;   long foo(int a1, int a2, int a3, int a4, struct t a5) {
;     return a1 + a2 + a3 + a4 + a5.a + a5.b;
;   }

define dso_local i64 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, [2 x i64] %4) local_unnamed_addr {
  %6 = extractvalue [2 x i64] %4, 0
  %7 = extractvalue [2 x i64] %4, 1
  %8 = add nsw i32 %1, %0
  %9 = add nsw i32 %8, %2
  %10 = add nsw i32 %9, %3
  %11 = sext i32 %10 to i64
  %12 = add nsw i64 %6, %11
  %13 = add nsw i64 %12, %7
  ret i64 %13
}
