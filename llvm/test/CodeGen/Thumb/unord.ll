; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s

define i32 @f1(float %X, float %Y) {
; CHECK-LABEL: _f1:
; CHECK: bne
; CHECK: .data_region
; CHECK: .long   ___unordsf2
	%tmp = fcmp uno float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}

define i32 @f2(float %X, float %Y) {
; CHECK-LABEL: _f2:
; CHECK: beq
; CHECK: .data_region
; CHECK: .long   ___unordsf2
	%tmp = fcmp ord float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}
