; Check that when removing arguments, incorrect callsite calling conventions are preserved

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; INTERESTING-LABEL: @fastcc_callee(
define fastcc i32 @fastcc_callee(i32 %a, i32 %b) {
  ret i32 %a
}

; INTERESTING-LABEL: @fastcc_callee_decl(
declare fastcc i32 @fastcc_callee_decl(i32 %a, i32 %b)

; INTERESTING-LABEL: @caller_wrong_callsites(
; INTERESTING: call
; INTERESTING: call

; RESULT-LABEL: define i32 @caller_wrong_callsites()
; RESULT: %call0 = call coldcc i32 @fastcc_callee()
; RESULT: %call1 = call i32 @fastcc_callee_decl()
define i32 @caller_wrong_callsites(i32 %x) {
  %call0 = call coldcc i32 @fastcc_callee(i32 %x, i32 2)
  %call1 = call ccc i32 @fastcc_callee_decl(i32 %x, i32 2)
  %result = add i32 %call0, %call1
  ret i32 %result
}

