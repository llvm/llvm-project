; REQUIRES: asserts
; RUN: opt -debug-only=verify-ipgo -passes='instcombine,instcombine' -verify-ipgo -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='instcombine,instcombine' -verify-ipgo -disable-output %s 2>&1 | FileCheck %s --check-prefix=VERIFY
;
; Verify cache invalidation is emitted for function- and module-level IR units.
;
; CHECK: PGOVerify cache invalidated: function
; CHECK: PGOVerify cache invalidated: module

; VERIFY: *** IPGO Verification After InstCombinePass ***
; VERIFY: *** IPGO Verification After InstCombinePass (Skipped) ***

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 0
  ret i32 %a
}
