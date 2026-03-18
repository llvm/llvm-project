; RUN: not llc -mtriple=bpf -mcpu=v3 < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function foo i64 (i32, i32, i32): {{(0x[0-9a-fA-F]+|t[0-9]+)}}: i64 = GlobalAddress<ptr @bar> 0 aggregate argument is split between registers and stack

; Source code:
;   struct t { long a; long b; };
;
;   long bar(int a1, int a2, int a3, int a4, struct t a5);
;   long foo(int a1, int a2, int a3) {
;     struct t tmp = {a1, a2};
;     return bar(a1, a2, a3, a2, tmp);
;   }

define dso_local i64 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = sext i32 %0 to i64
  %5 = sext i32 %1 to i64
  %6 = insertvalue [2 x i64] poison, i64 %4, 0
  %7 = insertvalue [2 x i64] %6, i64 %5, 1
  %8 = tail call i64 @bar(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %1, [2 x i64] %7)
  ret i64 %8
}

declare dso_local i64 @bar(i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64]) local_unnamed_addr
