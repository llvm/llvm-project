; RUN: llc < %s -mtriple=xcore | FileCheck %s

; Byte aligned store.
; CHECK-LABEL: align1:
; CHECK: bl __misaligned_store
define void @align1(ptr %p, i32 %val) nounwind {
entry:
	store i32 %val, ptr %p, align 1
	ret void
}

; Half word aligned store.
; CHECK: align2
; CHECK: st16
; CHECK: st16
define void @align2(ptr %p, i32 %val) nounwind {
entry:
	store i32 %val, ptr %p, align 2
	ret void
}
