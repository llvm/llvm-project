; RUN: opt < %s -passes=instcombine -verify-pgo-flow \
; RUN:     -verify-pgo-flow-fatal -disable-output 2>&1 | FileCheck %s
;
; No findings yet, so -verify-pgo-flow-fatal must not abort.

; CHECK: *** PGO Flow Verification After InstCombinePass ***{{$}}

define i32 @f(i32 %x) {
  %a = add i32 %x, 0
  ret i32 %a
}
