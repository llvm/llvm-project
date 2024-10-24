; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define signext i32 @foo() "function-instrument"="xray-always" {
; CHECK-LABEL: .Lxray_sled_0:
; CHECK:         j	.Ltmp[[#l:]]
; CHECK:         bcr	0, %r0
; CHECK:         llilf	%r2, 0
; CHECK:         brasl	%r14, __xray_FunctionEntry@PLT
; CHECK:       .Ltmp[[#l]]:
  ret i32 0
; CHECK-LABEL: .Lxray_sled_1:
; CHECK:         br	%r14
; CHECK:         bc	0, 0
; CHECK:         llilf	%r2, 0
; CHECK:         j	__xray_FunctionExit@PLT
}

; CHECK: 	.section	xray_instr_map,"ao",@progbits,foo
; CHECK: .Lxray_sleds_start0:
; CHECK: [[TMP1:.Ltmp[0-9]+]]:
; CHECK: 	.quad	.Lxray_sled_0-[[TMP1]]
; CHECK: 	.quad	.Lfunc_begin0-([[TMP1]]+8)
; CHECK: 	.byte	0x00
; CHECK: 	.byte	0x01
; CHECK: 	.byte	0x02
; CHECK:  .space	13
; CHECK: [[TMP2:.Ltmp[0-9]+]]:
; CHECK: 	.quad	.Lxray_sled_1-[[TMP2]]
; CHECK: 	.quad	.Lfunc_begin0-([[TMP2]]+8)
; CHECK: 	.byte	0x01
; CHECK: 	.byte	0x01
; CHECK: 	.byte	0x02
; CHECK: 	.space	13
; CHECK: .Lxray_sleds_end0:
; CHECK: 	.section	xray_fn_idx,"ao",@progbits,foo
; CHECK: 	.p2align	4
; CHECK: .Lxray_fn_idx0:
; CHECK: 	.quad	.Lxray_sleds_start0-.Lxray_fn_idx0
; CHECK: 	.quad	2
