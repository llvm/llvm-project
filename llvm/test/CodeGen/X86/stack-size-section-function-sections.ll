; RUN: llc < %s -mtriple=x86_64-linux -stack-size-section -function-sections | \
; RUN:   FileCheck %s --check-prefix=UNIQ
; RUN: llc < %s -mtriple=x86_64-linux -stack-size-section -function-sections -unique-section-names=0 | \
; RUN:   FileCheck %s --check-prefix=NOUNIQ

; Check we add SHF_LINK_ORDER for .stack_sizes and link it with the corresponding .text sections.
; UNIQ:   .section        .text._Z3barv,"ax",@progbits{{$}}
; UNIQ:   .section        .stack_sizes,"o",@progbits,.text._Z3barv{{$}}
; UNIQ:   .section        .text._Z3foov,"ax",@progbits{{$}}
; UNIQ:   .section        .stack_sizes,"o",@progbits,.text._Z3foov{{$}}
; NOUNIQ: .section        .text,"ax",@progbits,unique,1
; NOUNIQ: .section        .stack_sizes,"o",@progbits,.text,unique,1
; NOUNIQ: .section        .text,"ax",@progbits,unique,2
; NOUNIQ: .section        .stack_sizes,"o",@progbits,.text,unique,2

; Check we add .stack_size section to a COMDAT group with the corresponding .text section if such a COMDAT exists.
; UNIQ:   .section        .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat{{$}}
; UNIQ:   .section        .stack_sizes,"Go",@progbits,_Z4fooTIiET_v,comdat,.text._Z4fooTIiET_v{{$}}
; NOUNIQ: .section        .text,"axG",@progbits,_Z4fooTIiET_v,comdat,unique,3
; NOUNIQ: .section        .stack_sizes,"Go",@progbits,_Z4fooTIiET_v,comdat,.text,unique,3

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
