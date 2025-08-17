; Check that when removing arguments, existing bundles are preserved

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; INTERESTING-LABEL: @convergent_callee(
define i32 @convergent_callee(i32 %a, i32 %b) convergent {
  ret i32 %a
}

; INTERESTING-LABEL: @convergent_callee_decl(
declare i32 @convergent_callee_decl(i32 %a, i32 %b) convergent

; INTERESTING-LABEL: @convergent_caller(
; INTERESTING: call i32
; INTERESTING: call i32

; RESULT-LABEL: define i32 @convergent_caller()
; RESULT: %call0 = call i32 @convergent_callee() [ "convergencectrl"(token %entry.token) ]
; RESULT: %call1 = call i32 @convergent_callee_decl() [ "convergencectrl"(token %entry.token) ]
define i32 @convergent_caller(i32 %x) convergent {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %call0 = call i32 @convergent_callee(i32 %x, i32 2) [ "convergencectrl"(token %entry.token) ]
  %call1 = call i32 @convergent_callee_decl(i32 %x, i32 2) [ "convergencectrl"(token %entry.token) ]
  %result = add i32 %call0, %call1
  ret i32 %result
}

