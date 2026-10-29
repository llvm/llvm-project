; RUN: llc -mtriple=bpf -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-OFF-8 %s
; RUN: llc -mtriple=bpf -mcpu=v2 < %s | FileCheck --check-prefix=CHECK-OFF-8 %s
; RUN: llc -mtriple=bpf -mcpu=v3 < %s | FileCheck --check-prefix=CHECK-OFF-8 %s
; RUN: llc -mtriple=bpf -mcpu=v4 < %s | FileCheck --check-prefix=CHECK-OFF-8 %s
; RUN: llc -mtriple=bpf -mcpu=v1 < %s | FileCheck --check-prefix=CHECK-OFF-16 %s
; RUN: llc -mtriple=bpf -mcpu=v2 < %s | FileCheck --check-prefix=CHECK-OFF-16 %s
; RUN: llc -mtriple=bpf -mcpu=v3 < %s | FileCheck --check-prefix=CHECK-OFF-16 %s
; RUN: llc -mtriple=bpf -mcpu=v4 < %s | FileCheck --check-prefix=CHECK-OFF-16 %s

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

; The struct a5 is split: first half in r5, second half on stack.
; CHECK-LABEL:       foo:
; CHECK-OFF-8:       *(u64 *)(r11 - 8) = r[[#]]
; CHECK-OFF-16-NOT:  *(u64 *)(r11 - 16) = r[[#]]

declare dso_local i64 @bar(i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64]) local_unnamed_addr
