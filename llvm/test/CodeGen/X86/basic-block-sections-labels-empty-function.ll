;; Verify that the BB address map is not emitted for empty functions.
; RUN: llc < %s -mtriple=x86_64 -basic-block-sections=labels | FileCheck %s

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
; CHECK:		.quad	.Lfunc_begin1
