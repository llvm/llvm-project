; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; Each musttail call should fail to validate.

declare x86_stdcallcc void @cc_mismatch_callee()
define void @cc_mismatch() {
; CHECK: mismatched calling conv
  musttail call x86_stdcallcc void @cc_mismatch_callee()
  ret void
}

declare void @more_parms_callee(i32)
define void @more_parms() {
; CHECK: mismatched parameter counts
  musttail call void @more_parms_callee(i32 0)
  ret void
}

declare void @mismatched_intty_callee(i8)
define void @mismatched_intty(i32) {
; CHECK: mismatched parameter types
  musttail call void @mismatched_intty_callee(i8 0)
  ret void
}

declare void @mismatched_vararg_callee(i8*, ...)
define void @mismatched_vararg(i8*) {
; CHECK: mismatched varargs
  musttail call void (i8*, ...) @mismatched_vararg_callee(i8* null)
  ret void
}

; We would make this an implicit sret parameter, which would disturb the
; tail call.
declare { i32, i32, i32 } @mismatched_retty_callee(i32)
define void @mismatched_retty(i32) {
; CHECK: mismatched return types
  musttail call { i32, i32, i32 } @mismatched_retty_callee(i32 0)
  ret void
}

declare void @mismatched_byval_callee({ i32 }*)
define void @mismatched_byval({ i32 }* byval({ i32 }) %a) {
; CHECK: mismatched ABI impacting function attributes
  musttail call void @mismatched_byval_callee({ i32 }* %a)
  ret void
}

declare void @mismatched_inreg_callee(i32 inreg)
define void @mismatched_inreg(i32 %a) {
; CHECK: mismatched ABI impacting function attributes
  musttail call void @mismatched_inreg_callee(i32 inreg %a)
  ret void
}

declare void @mismatched_sret_callee(i32* sret(i32))
define void @mismatched_sret(i32* %a) {
; CHECK: mismatched ABI impacting function attributes
  musttail call void @mismatched_sret_callee(i32* sret(i32) %a)
  ret void
}

declare void @mismatched_alignment_callee(i32* byval(i32) align 8)
define void @mismatched_alignment(i32* byval(i32) align 4 %a) {
; CHECK: mismatched ABI impacting function attributes
  musttail call void @mismatched_alignment_callee(i32* byval(i32) align 8 %a)
  ret void
}

declare i32 @not_tail_pos_callee()
define i32 @not_tail_pos() {
; CHECK: musttail call must precede a ret with an optional bitcast
  %v = musttail call i32 @not_tail_pos_callee()
  %w = add i32 %v, 1
  ret i32 %w
}

define void @inline_asm() {
; CHECK: cannot use musttail call with inline asm
  musttail call void asm "ret", ""()
  ret void
}
