; RUN: llc < %s -mtriple=i686-win32 | FileCheck %s

; Previously we would forget to align to stack slot alignment after placing a
; byval argument.  Subsequent arguments would align themselves, but if it was
; the last argument, the argument size would not be a multiple of stack slot
; size. This resulted in retl $6 in callee-cleanup functions, as well as subtle
; varargs bugs.

%struct.Six = type { [6 x i8] }

define x86_stdcallcc void @f(%struct.Six* byval(%struct.Six) %a) {
  ret void
}
; CHECK-LABEL: _f@8:
; CHECK: retl $8

define x86_thiscallcc void @g(i8* %this, %struct.Six* byval(%struct.Six) %a) {
  ret void
}
; CHECK-LABEL: _g:
; CHECK: retl $8

define x86_fastcallcc void @h(i32 inreg %x, i32 inreg %y, %struct.Six* byval(%struct.Six) %a) {
  ret void
}
; CHECK-LABEL: @h@16:
; CHECK: retl $8
