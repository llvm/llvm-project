; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MODULE
; RUN: opt < %s -passes='function(verify-pgo-flow)' -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FUNCTION
; RUN: opt < %s -passes=instcombine -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOTRUN --allow-empty
; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-print-diagnostics=false -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=QUIET --allow-empty
;
; Named pipeline pass; does not need -verify-pgo-flow and does not change IR.

; MODULE: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; MODULE-NOT: (Skipped)

; FUNCTION: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; FUNCTION-NOT: (Skipped)

; NOTRUN-NOT: PGO Flow Verification

; QUIET-NOT: PGO Flow Verification

define i32 @f(i32 %x) {
  ret i32 %x
}
