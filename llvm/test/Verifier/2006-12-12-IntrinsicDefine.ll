; RUN: not llvm-as < %s 2>&1 | grep "llvm intrinsics cannot be defined"
; PR1047

define void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1) {
entry:
	ret void
}
