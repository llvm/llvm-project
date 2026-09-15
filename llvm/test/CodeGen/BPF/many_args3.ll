; RUN: llc < %s -mtriple=bpf -mcpu=v1 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v2 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v3 | FileCheck %s
; RUN: llc < %s -mtriple=bpf -mcpu=v4 | FileCheck %s

; Source code:
;   struct t { long a; long b; };
;   long foo1(int a1, int a2, int a3, int a4, int a5, short a6, long a7) {
;     return a1 + a2 + a3 + a4 + a5 + a6 + a7;
;   }
;
;   long foo2(int a1, int a2, int a3, int a4, int a5, struct t a6, int a7) {
;     return a1 + a2 + a3 + a4 + a5 + a6.a + a6.b + a7;
;   }
;
;   long foo3(struct t a1, int a2, int a3, int a4, int a5, struct t a6) {
;     return a1.a + a1.b + a2 + a3 + a4 + a5 + a6.a + a6.b;
;   }
;
;   long foo4(int a1, int a2, int a3, int a4, int a5, struct t a6, struct t a7) {
;     return a1 + a2 + a3 + a4 + a5 + a6.a + a6.b + a7.a + a7.b;
;   }
;
;   long bar5(int a1, int a2, int a3, int a4, int a5, short a6, long a7);
;   long foo5(int a1, int a2, int a3) {
;     return bar5(a1, a2, a3, a2, a3, a1, a2);
;   }
;
;   long bar6(int a1, int a2, int a3, int a4, int a5, struct t a6, int a7);
;   long foo6(int a1, int a2, int a3) {
;     struct t tmp = {a1, a2};
;     return bar6(a1, a2, a3, a2, a3, tmp, a2);
;   }
;
;   long bar7(struct t a1, int a2, int a3, int a4, int a5, struct t a6);
;   long foo7(int a1, int a2, int a3) {
;     struct t tmp1 = {a1, a2};
;     struct t tmp2 = {a2, a3};
;     return bar7(tmp1, a3, a2, a1, a2, tmp2);
;   }
;
;   long bar8(int a1, int a2, int a3, int a4, int a5, struct t a6, struct t a7);
;   long foo8(int a1, int a2, int a3) {
;     struct t tmp1 = { a3, a2 };
;     struct t  tmp2 = { a2, a3 };
;     return bar8(a1, a2, a3, a2, a3, tmp1, tmp2);
;   }

define dso_local i64 @foo1(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i16 noundef signext %5, i64 noundef %6) local_unnamed_addr {
  %8 = add nsw i32 %1, %0
  %9 = add nsw i32 %8, %2
  %10 = add nsw i32 %9, %3
  %11 = add nsw i32 %10, %4
  %12 = sext i16 %5 to i32
  %13 = add nsw i32 %11, %12
  %14 = sext i32 %13 to i64
  %15 = add nsw i64 %6, %14
  ret i64 %15
}

; CHECK-LABEL:   foo1:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)

define dso_local i64 @foo2(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, [2 x i64] %5, i32 noundef %6) local_unnamed_addr {
  %8 = extractvalue [2 x i64] %5, 0
  %9 = extractvalue [2 x i64] %5, 1
  %10 = add nsw i32 %1, %0
  %11 = add nsw i32 %10, %2
  %12 = add nsw i32 %11, %3
  %13 = add nsw i32 %12, %4
  %14 = sext i32 %13 to i64
  %15 = add nsw i64 %8, %14
  %16 = add nsw i64 %15, %9
  %17 = sext i32 %6 to i64
  %18 = add nsw i64 %16, %17
  ret i64 %18
}

; CHECK-LABEL:   foo2:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)
; CHECK:         r[[#]] = *(u64 *)(r11 + 24)

define dso_local i64 @foo3([2 x i64] %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, [2 x i64] %5) local_unnamed_addr {
  %7 = extractvalue [2 x i64] %0, 0
  %8 = extractvalue [2 x i64] %0, 1
  %9 = extractvalue [2 x i64] %5, 0
  %10 = extractvalue [2 x i64] %5, 1
  %11 = add nsw i64 %7, %8
  %12 = sext i32 %1 to i64
  %13 = add nsw i64 %11, %12
  %14 = sext i32 %2 to i64
  %15 = add nsw i64 %13, %14
  %16 = sext i32 %3 to i64
  %17 = add nsw i64 %15, %16
  %18 = sext i32 %4 to i64
  %19 = add nsw i64 %17, %18
  %20 = add nsw i64 %19, %9
  %21 = add nsw i64 %20, %10
  ret i64 %21
}

; CHECK-LABEL:   foo3:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)
; CHECK:         r[[#]] = *(u64 *)(r11 + 24)

define dso_local i64 @foo4(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, [2 x i64] %5, [2 x i64] %6) local_unnamed_addr {
  %8 = extractvalue [2 x i64] %5, 0
  %9 = extractvalue [2 x i64] %5, 1
  %10 = extractvalue [2 x i64] %6, 0
  %11 = extractvalue [2 x i64] %6, 1
  %12 = add nsw i32 %1, %0
  %13 = add nsw i32 %12, %2
  %14 = add nsw i32 %13, %3
  %15 = add nsw i32 %14, %4
  %16 = sext i32 %15 to i64
  %17 = add nsw i64 %8, %16
  %18 = add nsw i64 %17, %9
  %19 = add nsw i64 %18, %10
  %20 = add nsw i64 %19, %11
  ret i64 %20
}

; CHECK-LABEL:   foo4:
; CHECK:         r[[#]] = *(u64 *)(r11 + 8)
; CHECK:         r[[#]] = *(u64 *)(r11 + 16)
; CHECK:         r[[#]] = *(u64 *)(r11 + 24)
; CHECK:         r[[#]] = *(u64 *)(r11 + 32)

define dso_local i64 @foo5(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = trunc i32 %0 to i16
  %5 = sext i32 %1 to i64
  %6 = tail call i64 @bar5(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %1, i32 noundef %2, i16 noundef signext %4, i64 noundef %5)
  ret i64 %6
}

; CHECK-LABEL:   foo5:
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         *(u64 *)(r11 - 16) = r[[#]]

declare dso_local i64 @bar5(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i16 noundef signext, i64 noundef) local_unnamed_addr

define dso_local i64 @foo6(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = sext i32 %0 to i64
  %5 = sext i32 %1 to i64
  %6 = insertvalue [2 x i64] poison, i64 %4, 0
  %7 = insertvalue [2 x i64] %6, i64 %5, 1
  %8 = tail call i64 @bar6(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %1, i32 noundef %2, [2 x i64] %7, i32 noundef %1)
  ret i64 %8
}

; CHECK-LABEL:   foo6:
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         *(u64 *)(r11 - 16) = r[[#]]
; CHECK:         *(u64 *)(r11 - 24) = r[[#]]

declare dso_local i64 @bar6(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64], i32 noundef) local_unnamed_addr

define dso_local i64 @foo7(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = sext i32 %0 to i64
  %5 = sext i32 %1 to i64
  %6 = sext i32 %2 to i64
  %7 = insertvalue [2 x i64] poison, i64 %4, 0
  %8 = insertvalue [2 x i64] %7, i64 %5, 1
  %9 = insertvalue [2 x i64] poison, i64 %5, 0
  %10 = insertvalue [2 x i64] %9, i64 %6, 1
  %11 = tail call i64 @bar7([2 x i64] %8, i32 noundef %2, i32 noundef %1, i32 noundef %0, i32 noundef %1, [2 x i64] %10)
  ret i64 %11
}

; CHECK-LABEL:   foo7:
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         *(u64 *)(r11 - 16) = r[[#]]
; CHECK:         *(u64 *)(r11 - 24) = r[[#]]

declare dso_local i64 @bar7([2 x i64], i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64]) local_unnamed_addr

define dso_local i64 @foo8(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = sext i32 %2 to i64
  %5 = sext i32 %1 to i64
  %6 = insertvalue [2 x i64] poison, i64 %4, 0
  %7 = insertvalue [2 x i64] %6, i64 %5, 1
  %8 = insertvalue [2 x i64] poison, i64 %5, 0
  %9 = insertvalue [2 x i64] %8, i64 %4, 1
  %10 = tail call i64 @bar8(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %1, i32 noundef %2, [2 x i64] %7, [2 x i64] %9)
  ret i64 %10
}

; CHECK-LABEL:   foo8:
; CHECK:         *(u64 *)(r11 - 8) = r[[#]]
; CHECK:         *(u64 *)(r11 - 16) = r[[#]]
; CHECK:         *(u64 *)(r11 - 24) = r[[#]]
; CHECK:         *(u64 *)(r11 - 32) = r[[#]]

; CHECK-NOT:     *(u64 *)(r11 - 40) = r[[#]]

declare dso_local i64 @bar8(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64], [2 x i64]) local_unnamed_addr
