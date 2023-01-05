; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: llvm intrinsics cannot be defined
; PR1047

define void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1) {
entry:
	ret void
}
