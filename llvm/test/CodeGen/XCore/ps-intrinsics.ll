; RUN: llc < %s -march=xcore | FileCheck %s
declare i32 @llvm.xcore.getps(i32)
declare void @llvm.xcore.setps(i32, i32)

define i32 @getps(i32 %reg) nounwind {
; CHECK-LABEL: getps:
; CHECK: get r0, ps[r0]
	%result = call i32 @llvm.xcore.getps(i32 %reg)
	ret i32 %result
}


define void @setps(i32 %reg, i32 %value) nounwind {
; CHECK-LABEL: setps:
; CHECK: set ps[r0], r1
	call void @llvm.xcore.setps(i32 %reg, i32 %value)
	ret void
}
