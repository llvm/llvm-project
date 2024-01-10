; RUN: llc < %s -mtriple=x86_64-linux -stack-size-section | FileCheck %s --check-prefix=CHECK --check-prefix=GROUPS

; PS4 'as' does not recognize the section attribute "o".  So we have a simple .stack_sizes section on PS4.
; RUN: llc < %s -mtriple=x86_64-scei-ps4 -stack-size-section | FileCheck %s --check-prefix=CHECK --check-prefix=NOGROUPS

; CHECK-LABEL: func1:
; CHECK-NEXT: .Lfunc_begin0:
; GROUPS: .section .stack_sizes,"o",@progbits,.text{{$}}
; NOGROUPS: .section .stack_sizes,"",@progbits
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .byte 8
define void @func1(i32, i32) #0 {
  alloca i32, align 4
  alloca i32, align 4
  ret void
}

; CHECK-LABEL: func2:
; CHECK-NEXT: .Lfunc_begin1:
; GROUPS: .section .stack_sizes,"o",@progbits,.text{{$}}
; NOGROUPS: .section .stack_sizes,"",@progbits
; CHECK-NEXT: .quad .Lfunc_begin1
; CHECK-NEXT: .byte 24
define void @func2() #0 {
  alloca i32, align 4
  call void @func1(i32 1, i32 2)
  ret void
}

; Check that we still put .stack_sizes into the corresponding COMDAT group if any.
; CHECK: .section .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat
; GROUPS: .section .stack_sizes,"oG",@progbits,.text._Z4fooTIiET_v,_Z4fooTIiET_v,comdat{{$}}
; NOGROUPS: .section .stack_sizes,"",@progbits
$_Z4fooTIiET_v = comdat any
define linkonce_odr dso_local i32 @_Z4fooTIiET_v() comdat {
  ret i32 0
}

; CHECK: .section .text.func3,"ax",@progbits
; GROUPS: .section .stack_sizes,"o",@progbits,.text.func3{{$}}
; NOGROUPS: .section .stack_sizes,"",@progbits
define dso_local i32 @func3() section ".text.func3" {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  ret i32 0
}

; CHECK-LABEL: dynalloc:
; CHECK-NOT: .section .stack_sizes
define void @dynalloc(i32 %N) #0 {
  alloca i32, i32 %N
  ret void
}

attributes #0 = { "frame-pointer"="all" }
