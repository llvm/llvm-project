; RUN: llc < %s -mtriple=x86_64 -function-sections -func-map | FileCheck %s

$_Z4fooTIiET_v = comdat any

define dso_local i32 @_Z3barv() {
  ret i32 0
}

;; Check we add SHF_LINK_ORDER for .llvm_func_map and link it with the corresponding .text sections.
; CHECK:		.section .text._Z3barv,"ax",@progbits
; CHECK-LABEL:	_Z3barv:
; CHECK-NEXT:	[[BAR_BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:		.section .llvm_func_map,"Mo",@llvm_func_map,17,.text._Z3barv{{$}}
; CHECK-NEXT:		.byte 1			# version
; CHECK-NEXT:		.quad [[BAR_BEGIN]]	# function address

define dso_local i32 @_Z3foov() {
  %1 = call i32 @_Z4fooTIiET_v()
  ret i32 %1
}

; CHECK:		.section .text._Z3foov,"ax",@progbits
; CHECK-LABEL:	_Z3foov:
; CHECK-NEXT:	[[FOO_BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:		.section  .llvm_func_map,"Mo",@llvm_func_map,17,.text._Z3foov{{$}}
; CHECK-NEXT:		.byte 1			# version
; CHECK-NEXT:		.quad [[FOO_BEGIN]]	# function address


define linkonce_odr dso_local i32 @_Z4fooTIiET_v() comdat {
  ret i32 0
}

;; Check we add .llvm_func_map section to a COMDAT group with the corresponding .text section if such a COMDAT exists.
; CHECK:		.section .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat
; CHECK-LABEL:	_Z4fooTIiET_v:
; CHECK-NEXT:	[[FOOCOMDAT_BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:		.section .llvm_func_map,"MoG",@llvm_func_map,17,.text._Z4fooTIiET_v,_Z4fooTIiET_v,comdat{{$}}
; CHECK-NEXT:		.byte 1				# version
; CHECK-NEXT:		.quad [[FOOCOMDAT_BEGIN]]	# function address
