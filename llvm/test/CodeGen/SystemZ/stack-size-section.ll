; RUN: llc < %s -mtriple=s390x-linux-gnu -stack-size-section | FileCheck %s

; CHECK-LABEL: func1:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .byte 0
define void @func1(i32, i32) #0 {
  ret void
}

; CHECK-LABEL: func2:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin1
; CHECK-NEXT: .ascii  "\250\001"
define void @func2(i32, i32) #0 {
  alloca i32, align 4
  alloca i32, align 4
  ret void
}

; CHECK-LABEL: func3:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin2
; CHECK-NEXT: .ascii  "\250\001"
define void @func3() #0 {
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

; Check that .stack_sizes section is linked to the function's section (.text),
; and not to the section containing the jump table (.rodata).
; CHECK-LABEL: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin4
; CHECK-NEXT: .ascii "\260!"
define i32 @linked_section(i32 %x) {
  %arr = alloca [1024 x i32]
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb0:
  ret i32 0

sw.bb1:
  ret i32 1

sw.bb2:
  ret i32 2

sw.bb3:
  ret i32 3

sw.epilog:
  ret i32 -1
}

attributes #0 = { "frame-pointer"="all" }
