; RUN: llc < %s -mtriple=x86_64-linux -stack-size-section -function-sections | FileCheck %s

; Check we add SHF_LINK_ORDER for .stack_sizes and link it with the corresponding .text sections.
; CHECK: .section        .text._Z3barv,"ax",@progbits
; CHECK: .section        .stack_sizes,"o",@progbits,.text._Z3barv,unique,0
; CHECK: .section        .text._Z3foov,"ax",@progbits
; CHECK: .section        .stack_sizes,"o",@progbits,.text._Z3foov,unique,1

; Check we add .stack_size section to a COMDAT group with the corresponding .text section if such a COMDAT exists.
; CHECK: .section        .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat
; CHECK: .section        .stack_sizes,"Go",@progbits,_Z4fooTIiET_v,comdat,.text._Z4fooTIiET_v,unique,2

$_Z4fooTIiET_v = comdat any

define dso_local i32 @_Z3barv() {
  ret i32 0
}

define dso_local i32 @_Z3foov() {
  %1 = call i32 @_Z4fooTIiET_v()
  ret i32 %1
}

define linkonce_odr dso_local i32 @_Z4fooTIiET_v() comdat {
  ret i32 0
}
