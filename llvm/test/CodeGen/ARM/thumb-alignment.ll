; RUN: llc -relocation-model=pic -mtriple=thumbv7-unknown-linux -o - %s | FileCheck %s

@x = external global i32

; CHECK: .globl	foo
; CHECK-NEXT: .align	2
define i32* @foo() {
  ret i32* @x
}

; CHECK: .globl	bar
; CHECK-NEXT: .align	1
define i32* @bar() {
  ret i32* zeroinitializer
}

@a = external global i32
@b = external global i32
@c = external global i32
@d = external global i32

; Create a Thumb-2 jump table, which should force alignment to 4 bytes.

; CHECK: .globl	baz
; CHECK-NEXT: .align	2
; CHECK: tbb
define i32 @baz() {
  %1 = load i32, i32* @c, align 4
  switch i32 %1, label %7 [
    i32 1, label %2
    i32 4, label %5
    i32 9, label %5
    i32 3, label %8
  ]

; <label>:2
  %3 = load i32, i32* @a, align 4
  %4 = tail call i32 bitcast (i32 (...)* @fn2 to i32 (i32 (...)*, i32, i32)*)(i32 (...)* bitcast (i32 ()* @baz to i32 (...)*), i32 0, i32 %3) #2
  br label %8

; <label>:5
  %6 = load i32, i32* @d, align 4
  store i32 %6, i32* @b, align 4
  br label %8

; <label>:7
  br label %8

; <label>:8
  %e.0 = phi i32 [ 1, %7 ], [ 1, %2 ], [ 0, %0 ], [ 0, %5 ]
  ret i32 %e.0
}

declare i32 @fn2(...)
