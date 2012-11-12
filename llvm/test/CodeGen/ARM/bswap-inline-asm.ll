; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6 | FileCheck %s

define i32 @t1(i32 %x) nounwind {
; CHECK: t1:
; CHECK-NOT: InlineAsm
; CHECK: rev
  %asmtmp = tail call i32 asm "rev $0, $1\0A", "=l,l"(i32 %x) nounwind
  ret i32 %asmtmp
}
