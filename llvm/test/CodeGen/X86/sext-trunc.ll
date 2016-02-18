; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s

define signext i8 @foo(i16 signext  %x) nounwind  {
	%retval56 = trunc i16 %x to i8
	ret i8 %retval56

; CHECK-LABEL: foo:
; CHECK: movb
; CHECK-NEXT: retl
}
