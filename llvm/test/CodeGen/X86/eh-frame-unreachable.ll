; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; Test that we don't emit a row that extends beyond the FDE's range_size.
;
; CHECK: movq	%rsp, %rbp
; CHECK-NEXT:	.cfi_endproc
; CHECK-NOT: .cfi

define void @f() #0 {
  unreachable
}
attributes #0 = { "no-frame-pointer-elim"="true" }
