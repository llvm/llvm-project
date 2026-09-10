; RUN: opt < %s -passes=instcombine -verify-pgo-flow \
; RUN:     -verify-pgo-flow-fatal -disable-output 2>&1 | FileCheck %s
;
; No InstrProf summary and no mismatch, so -verify-pgo-flow-fatal must not abort.
; Skip notes are covered in verify-pgo-flow-approx-profile.ll and
; verify-pgo-flow-count-overflow-skip.ll, a real mismatch abort is in
; verify-pgo-flow-block-frequency.ll.

; CHECK: *** PGO Flow Verification After InstCombinePass ***{{$}}

define i32 @f(i32 %x) {
  %a = add i32 %x, 0
  ret i32 %a
}
