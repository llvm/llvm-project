; RUN: llc < %s -mtriple=armv7-linux -stack-size-section | FileCheck %s

; CHECK-LABEL: func1:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK: .section .stack_sizes,"o",%progbits,.text{{$}}
; CHECK-NEXT: .long .Lfunc_begin0
; CHECK-NEXT: .byte 8
define void @func1(i32, i32) #0 {
  alloca i32, align 4
  alloca i32, align 4
  ret void
}

; CHECK-LABEL: func2:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK: .section .stack_sizes,"o",%progbits,.text{{$}}
; CHECK-NEXT: .long .Lfunc_begin1
; CHECK-NEXT: .byte 16
define void @func2() #0 {
  alloca i32, align 4
  call void @func1(i32 1, i32 2)
  ret void
}

; CHECK-LABEL: dynalloc:
; CHECK-NOT: .section .stack_sizes
define void @dynalloc(i32 %N) #0 {
  alloca i32, i32 %N
  ret void
}

attributes #0 = { "frame-pointer"="all" }
