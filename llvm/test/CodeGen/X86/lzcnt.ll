; RUN: llc < %s -mtriple=x86_64-- -mattr=+lzcnt | FileCheck %s

declare i8 @llvm.ctlz.i8(i8, i1) nounwind readnone
declare i16 @llvm.ctlz.i16(i16, i1) nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i8 @t1(i8 %x) nounwind  {
	%tmp = tail call i8 @llvm.ctlz.i8( i8 %x, i1 false )
	ret i8 %tmp
; CHECK-LABEL: t1:
; CHECK: lzcntl
}

define i16 @t2(i16 %x) nounwind  {
	%tmp = tail call i16 @llvm.ctlz.i16( i16 %x, i1 false )
	ret i16 %tmp
; CHECK-LABEL: t2:
; CHECK: lzcntw
}

define i32 @t3(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x, i1 false )
	ret i32 %tmp
; CHECK-LABEL: t3:
; CHECK: lzcntl
}

define i64 @t4(i64 %x) nounwind  {
	%tmp = tail call i64 @llvm.ctlz.i64( i64 %x, i1 false )
	ret i64 %tmp
; CHECK-LABEL: t4:
; CHECK: lzcntq
}

define i8 @t5(i8 %x) nounwind  {
	%tmp = tail call i8 @llvm.ctlz.i8( i8 %x, i1 true )
	ret i8 %tmp
; CHECK-LABEL: t5:
; CHECK: lzcntl
}

define i16 @t6(i16 %x) nounwind  {
	%tmp = tail call i16 @llvm.ctlz.i16( i16 %x, i1 true )
	ret i16 %tmp
; CHECK-LABEL: t6:
; CHECK: lzcntw
}

define i32 @t7(i32 %x) nounwind  {
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x, i1 true )
	ret i32 %tmp
; CHECK-LABEL: t7:
; CHECK: lzcntl
}

define i64 @t8(i64 %x) nounwind  {
	%tmp = tail call i64 @llvm.ctlz.i64( i64 %x, i1 true )
	ret i64 %tmp
; CHECK-LABEL: t8:
; CHECK: lzcntq
}
