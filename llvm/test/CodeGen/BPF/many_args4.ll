; RUN: llc < %s -mtriple=bpf -mcpu=v1 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v2 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v3 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v4 | FileCheck %s

; Source code:
;   __attribute__((noinline)) long foo1(int a, int b, int c, int d, int e, int f) {
;     return a + b + c + d + e + f;
;   }
;
;  __attribute__((noinline)) long foo2(int a, int b, int c, int d, int e, int f, int g) {
;    return a + b + c + d + e + f + g;
;  }
;
;  long bar(int a, int b, int c, int d, int e, int f, int g) {
;    long r1 = foo1(a, b, c, d, e, f + g);
;    long r2 = foo2(a, b, c, d, e, f, g);
;    return r1 + r2;
;  }


define dso_local range(i64 -2147483648, 2147483648) i64 @foo1(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr {
  %7 = add nsw i32 %1, %0
  %8 = add nsw i32 %7, %2
  %9 = add nsw i32 %8, %3
  %10 = add nsw i32 %9, %4
  %11 = add nsw i32 %10, %5
  %12 = sext i32 %11 to i64
  ret i64 %12
}

; CHECK-LABEL:   foo1:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)

define dso_local range(i64 -2147483648, 2147483648) i64 @foo2(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) local_unnamed_addr {
  %8 = add nsw i32 %1, %0
  %9 = add nsw i32 %8, %2
  %10 = add nsw i32 %9, %3
  %11 = add nsw i32 %10, %4
  %12 = add nsw i32 %11, %5
  %13 = add nsw i32 %12, %6
  %14 = sext i32 %13 to i64
  ret i64 %14
}

; CHECK-LABEL:   foo2:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)

define dso_local range(i64 -4294967296, 4294967295) i64 @bar(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) local_unnamed_addr {
  %8 = add nsw i32 %6, %5
  %9 = tail call i64 @foo1(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %8)
  %10 = tail call i64 @foo2(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6)
  %11 = add nsw i64 %10, %9
  ret i64 %11
}

; CHECK-LABEL:   bar:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         call foo1
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         *(u64 *)(r11 - 16) = r[[#]]
; CHECK:         call foo2
