; RUN: llc < %s -mtriple=i386-apple-darwin  | FileCheck %s


define internal i64 @baz() nounwind {
  %tmp = load i64* @"+x"
  ret i64 %tmp
; CHECK: _baz:
; CHECK:    movl "L_+x$non_lazy_ptr", %ecx
}


@"+x" = external global i64

; CHECK: "L_+x$non_lazy_ptr":
; CHECK:	.indirect_symbol "_+x"
