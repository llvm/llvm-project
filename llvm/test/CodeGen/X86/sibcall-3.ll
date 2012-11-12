; RUN: llc < %s -mtriple=i386-unknown-unknown | FileCheck %s
; PR7193

define void @t1(i8* inreg %dst, i8* inreg %src, i8* inreg %len) nounwind {
; CHECK: t1:
; CHECK: calll 0
  tail call void null(i8* inreg %dst, i8* inreg %src, i8* inreg %len) nounwind
  ret void
}

define void @t2(i8* inreg %dst, i8* inreg %src, i8* inreg %len) nounwind {
; CHECK: t2:
; CHECK: jmpl
  tail call void null(i8* inreg %dst, i8* inreg %src) nounwind
  ret void
}
