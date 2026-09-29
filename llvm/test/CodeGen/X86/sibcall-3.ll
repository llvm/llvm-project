; RUN: llc < %s -mtriple=i386-unknown-unknown | FileCheck %s
; PR7193

define void @t1(ptr inreg %dst, ptr inreg %src, ptr inreg %len) nounwind {
; CHECK-LABEL: t1:
; CHECK: calll 0
  tail call void null(ptr inreg %dst, ptr inreg %src, ptr inreg %len) nounwind
  ret void
}

define void @t2(ptr inreg %dst, ptr inreg %src, ptr inreg %len) nounwind {
; CHECK-LABEL: t2:
; CHECK: jmpl
  tail call void null(ptr inreg %dst, ptr inreg %src) nounwind
  ret void
}
