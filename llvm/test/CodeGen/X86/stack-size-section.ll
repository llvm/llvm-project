; RUN: llc < %s -mtriple=x86_64-linux -stack-size-section | FileCheck %s

; CHECK-LABEL: func1:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .byte 8
define void @func1(i32, i32) #0 {
  alloca i32, align 4
  alloca i32, align 4
  ret void
}

; CHECK-LABEL: func2:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .quad .Lfunc_begin1
; CHECK-NEXT: .byte 24
define void @func2() #0 {
  alloca i32, align 4
  call void @func1(i32 1, i32 2)
  ret void
}

; Check that we still put .stack_sizes into the corresponding COMDAT group if any.
; CHECK: .section .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat
; CHECK: .section .stack_sizes,"Go",@progbits,_Z4fooTIiET_v,comdat,.text._Z4fooTIiET_v{{$}}
$_Z4fooTIiET_v = comdat any
define linkonce_odr dso_local i32 @_Z4fooTIiET_v() comdat {
  ret i32 0
}

; CHECK: .section .text.func3,"ax",@progbits
; CHECK: .section .stack_sizes,"o",@progbits,.text.func3{{$}}
define dso_local i32 @func3() section ".text.func3" {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 0
}

; CHECK-LABEL: dynalloc:
; CHECK-NOT: .section .stack_sizes
define void @dynalloc(i32 %N) #0 {
  alloca i32, i32 %N
  ret void
}

attributes #0 = { "frame-pointer"="all" }
