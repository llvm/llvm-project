; RUN: not llvm-as < %s |& grep {llvm intrinsics cannot be defined}
; PR1047

define void @llvm.memcpy.i32(i8*, i8*, i32, i32) {
entry:
	ret void
}
