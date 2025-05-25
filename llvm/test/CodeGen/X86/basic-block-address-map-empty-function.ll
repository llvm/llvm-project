;; Verify that the BB address map is not emitted for empty functions.
; RUN: llc < %s -mtriple=x86_64 -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,BASIC
; RUN: llc < %s -mtriple=x86_64 -basic-block-address-map -pgo-analysis-map=func-entry-count,bb-freq | FileCheck %s --check-prefixes=CHECK,PGO

define void @empty_func() {
entry:
  unreachable
}
; CHECK:		{{^ *}}.text{{$}}
; CHECK:	empty_func:
; CHECK:	.Lfunc_begin0:
; CHECK-NOT:	.section	.llvm_bb_addr_map

define void @func() {
entry:
  ret void
}

; CHECK:	func:
; CHECK:	.Lfunc_begin1:
; CHECK:		.section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text{{$}}
; CHECK-NEXT:		.byte 2			# version
; BASIC-NEXT:		.byte 0			# feature
; PGO-NEXT:		.byte 3			# feature
; CHECK-NEXT:		.quad	.Lfunc_begin1	# function address
