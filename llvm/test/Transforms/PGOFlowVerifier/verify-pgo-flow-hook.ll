; RUN: opt < %s -passes=instcombine -verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ON
; RUN: opt < %s -passes=instcombine -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OFF --allow-empty
; RUN: opt < %s -passes='instcombine,instcombine' -verify-pgo-flow \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=TWICE
; RUN: opt < %s -passes=instcombine -verify-pgo-flow \
; RUN:     -verify-pgo-flow-print-diagnostics=false -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=QUIET --allow-empty
; RUN: opt < %s -passes='loop(indvars)' -verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOOP
;
; Off unless -verify-pgo-flow. Unchanged passes are skipped. Loop adaptors
; are not walked.

; ON: *** PGO Flow Verification After InstCombinePass ***{{$}}

; OFF-NOT: PGO Flow Verification

; TWICE: *** PGO Flow Verification After InstCombinePass ***{{$}}
; TWICE: *** PGO Flow Verification After InstCombinePass (Skipped) ***

; QUIET-NOT: PGO Flow Verification

; LOOP-NOT: PassAdaptor
; LOOP: After IndVarSimplifyPass

define i32 @f(i32 %x) {
  %a = add i32 %x, 0
  ret i32 %a
}

define void @loop(ptr %p, i32 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
  %slot = getelementptr i32, ptr %p, i32 %i
  store i32 %i, ptr %slot
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}
