; BB cluster func cfg node num tests for v0.
;
;; Profile for version 0 check node num:
; RUN: echo '!foo' > %t1
; RUN: echo '#correct node_count given' >> %t1
; RUN: echo '$node_count 14' >> %t1
; RUN: echo '!!0 5 6 18 3 4 8 9 11 15 16 12 13' >> %t1
; RUN: echo '!bar' >> %t1
; RUN: echo '#wrong node_count given' >> %t1
; RUN: echo '$node_count 4' >> %t1
; RUN: echo '!!0 1 2' >> %t1
; RUN: echo '!main' >> %t1

; RUN: llc -O0 --basic-block-sections=%t1 -o %t %s  2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
; CHECK-WARNING: warning: MF bar: node count mismatch (profile=4 actual=5)

define dso_local void @bar() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store volatile i32 0, i32* %1, align 4
  store i32 0, i32* %2, align 4
  br label %3

; <label>:3:                                      ; preds = %10, %0
  %4 = load i32, i32* %2, align 4
  %5 = icmp slt i32 %4, 1000
  br i1 %5, label %6, label %13

; <label>:6:                                      ; preds = %3
  %7 = load i32, i32* %2, align 4
  %8 = load volatile i32, i32* %1, align 4
  %9 = add nsw i32 %8, %7
  store volatile i32 %9, i32* %1, align 4
  br label %10

; <label>:10:                                     ; preds = %6
  %11 = load i32, i32* %2, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, i32* %2, align 4
  br label %3

; <label>:13:                                     ; preds = %3
  ret void
}

define dso_local void @foo(i32) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  store i32 2147483640, i32* %3, align 4
  store i32 0, i32* %4, align 4
  br label %6

; <label>:6:                                      ; preds = %30, %1
  %7 = load i32, i32* %4, align 4
  %8 = load i32, i32* %2, align 4
  %9 = icmp slt i32 %7, %8
  br i1 %9, label %10, label %33

; <label>:10:                                     ; preds = %6
  %11 = load i32, i32* %3, align 4
  %12 = srem i32 %11, 100
  store i32 %12, i32* %5, align 4
  %13 = load i32, i32* %5, align 4
  %14 = icmp slt i32 %13, 90
  br i1 %14, label %15, label %21

; <label>:15:                                     ; preds = %10
  %16 = load i32, i32* %5, align 4
  %17 = icmp slt i32 %16, 70
  br i1 %17, label %18, label %19

; <label>:18:                                     ; preds = %15
  call void @bar()
  br label %20

; <label>:19:                                     ; preds = %15
  call void @bar()
  br label %20

; <label>:20:                                     ; preds = %19, %18
  br label %27

; <label>:21:                                     ; preds = %10
  %22 = load i32, i32* %5, align 4
  %23 = icmp slt i32 %22, 93
  br i1 %23, label %24, label %25

; <label>:24:                                     ; preds = %21
  call void @bar()
  br label %26

; <label>:25:                                     ; preds = %21
  call void @bar()
  br label %26

; <label>:26:                                     ; preds = %25, %24
  br label %27

; <label>:27:                                     ; preds = %26, %20
  %28 = load i32, i32* %3, align 4
  %29 = add nsw i32 %28, -1
  store i32 %29, i32* %3, align 4
  br label %30

; <label>:30:                                     ; preds = %27
  %31 = load i32, i32* %4, align 4
  %32 = add nsw i32 %31, 1
  store i32 %32, i32* %4, align 4
  br label %6

; <label>:33:                                     ; preds = %6
  ret void
}
