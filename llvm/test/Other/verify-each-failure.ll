; A test that the option -verify-each reports the last pass run
; when a failure occurs.

; RUN: not --crash opt -disable-output -debug-pass-manager -verify-each -passes="module(verify-each-failure)" %s 2>&1 | FileCheck %s --check-prefix=CHECK_MODULE
; RUN: not --crash opt -disable-output -debug-pass-manager -verify-each -passes="function(verify-each-failure)" %s 2>&1 | FileCheck %s --check-prefix=CHECK_FUNCTION

; CHECK_MODULE: Running pass: TriggerVerifierFailurePass on [module]
; CHECK_MODULE: Broken module found after pass "TriggerVerifierFailurePass", compilation aborted!

; CHECK_FUNCTION: Running pass: TriggerVerifierFailurePass on main
; CHECK_FUNCTION: Broken function found after pass "TriggerVerifierFailurePass", compilation aborted!

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}
