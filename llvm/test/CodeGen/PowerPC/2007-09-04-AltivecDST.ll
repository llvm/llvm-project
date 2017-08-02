; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mattr=+altivec | grep dst | count 4

define hidden void @_Z4borkPc(i8* %image) {
entry:
	tail call void @llvm.ppc.altivec.dst( i8* %image, i32 8, i32 0 )
	tail call void @llvm.ppc.altivec.dstt( i8* %image, i32 8, i32 0 )
	tail call void @llvm.ppc.altivec.dstst( i8* %image, i32 8, i32 0 )
	tail call void @llvm.ppc.altivec.dststt( i8* %image, i32 8, i32 0 )
	ret void
}

declare void @llvm.ppc.altivec.dst(i8*, i32, i32)
declare void @llvm.ppc.altivec.dstt(i8*, i32, i32)
declare void @llvm.ppc.altivec.dstst(i8*, i32, i32)
declare void @llvm.ppc.altivec.dststt(i8*, i32, i32)
