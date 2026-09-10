; REQUIRES: asserts
; RUN: opt < %s -passes=instcombine -verify-pgo-flow \
; RUN:     -debug-only=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; No ProfileSummary: walk the function, then skip InstrProf use-phase checks.

; CHECK: *** PGO Flow Verification After InstCombinePass ***{{$}}
; CHECK: PGOFlowVerifier: skip 'f' (no InstrProf use-phase summary)

define i32 @f(i32 %x) {
  %a = add i32 %x, 0
  ret i32 %a
}
