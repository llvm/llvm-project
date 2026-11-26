; RUN: llc < %s -mtriple=bpf -mcpu=v1 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v2 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v3 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v4 | FileCheck --check-prefix=CHECK-V4 %s

; Source code:
;   long foo(int, int, int, int, int, long);
;   long bar(int a, int b, int c, int d, int e) {
;     return foo(a, b, c, d, e, 16) + foo(a, b, c, d, e, 0xffFFffFF);
;   }

define dso_local i64 @bar(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr {
  %6 = tail call i64 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i64 noundef 16)
  %7 = tail call i64 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i64 noundef 4294967295)
  %8 = add nsw i64 %7, %6
  ret i64 %8
}

declare dso_local i64 @foo(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef) local_unnamed_addr

; CHECK-LABEL:   bar:
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK-V4:      *(u64 *)(r11 - 8) = 16
; CHECK:         call foo
; CHECK-V4:      call foo
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK-V4:      *(u64 *)(r11 - 8) = r[[#]]
