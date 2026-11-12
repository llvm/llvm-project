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

; The struct a5 is split: first half in r5, second half on stack.
; CHECK-LABEL:       foo:
; CHECK-OFF-8:       r[[#]] = *(u64 *)(r11 + 8)
; CHECK-OFF-16-NOT:  r[[#]] = *(u64 *)(r11 + 16)
