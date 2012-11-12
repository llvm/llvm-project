; RUN: llc < %s -march=xcore | FileCheck %s
declare void @llvm.xcore.setsr(i32)
declare void @llvm.xcore.clrsr(i32)

define void @setsr() nounwind {
; CHECK: setsr:
; CHECK: setsr 128
	call void @llvm.xcore.setsr(i32 128)
	ret void
}


define void @clrsr() nounwind {
; CHECK: clrsr:
; CHECK: clrsr 128
	call void @llvm.xcore.clrsr(i32 128)
	ret void
}
