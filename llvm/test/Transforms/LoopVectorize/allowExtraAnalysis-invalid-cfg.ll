; RUN: opt -S -passes=loop-vectorize -pass-remarks-output=%t.yaml < %s
; RUN: FileCheck --input-file=%t.yaml --check-prefix=REMARKS %s

; Enabling remarks forces extra legality analysis on loops with unexpected CFG.
; Ensure LV does not crash due to the loop having multiple latches.
; A normal branch gets canonicalized to a unique latch before LV runs, so use
; indirectbr to keep the invalid CFG shape.

; This files contains unsupported CFG cases that can cause crashes with DoExtraAnalysis:
; - A loop with multiple latches
; - A loop with multiple latches and predecessors
; - A loop with multiple predecessors and a conditional block


; REMARKS:      Name:            CFGNotUnderstood
; REMARKS-NEXT: Function:        multiple_latches_indirectbr

; REMARKS:      Name:            NoLatchEarlyExit
; REMARKS-NEXT: Function:        multiple_latches_indirectbr

; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-vectorize
; REMARKS-NEXT: Name:            MissedDetails
; REMARKS-NEXT: Function:        multiple_latches_indirectbr
; REMARKS-NEXT: Args:
; REMARKS-NEXT:   - String:          loop not vectorized
; REMARKS-NEXT: ...
define void @multiple_latches_indirectbr(ptr %ptrA, ptr %ptrB) {
entry:
  indirectbr ptr %ptrA, [label %loop]

loop:
  br i1 false, label %bb, label %loop

bb:
  br label %loop
}

; REMARKS:      Name:            CFGNotUnderstood
; REMARKS-NEXT: Function:        multiple_latches_and_predecessors_indirectbr

; REMARKS:      Name:            NoLatchEarlyExit
; REMARKS-NEXT: Function:        multiple_latches_and_predecessors_indirectbr

; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-vectorize
; REMARKS-NEXT: Name:            MissedDetails
; REMARKS-NEXT: Function:        multiple_latches_and_predecessors_indirectbr
; REMARKS-NEXT: Args:
; REMARKS-NEXT:   - String:          loop not vectorized
; REMARKS-NEXT: ...
define void @multiple_latches_and_predecessors_indirectbr(ptr %ptrA, ptr %ptrB) {
entry:
  indirectbr ptr %ptrA, [label %loop, label %side]

side:
  br label %loop

loop:
  %x = load i64, ptr %ptrB, align 8
  br i1 false, label %back, label %loop

back:
  br label %loop
}

; REMARKS:      Name:            CFGNotUnderstood
; REMARKS-NEXT: Function:        multiple_predecessors_cond_block

; REMARKS:      Name:            UnsupportedUncountableLoop
; REMARKS-NEXT: Function:        multiple_predecessors_cond_block

; REMARKS: --- !Missed
; REMARKS-NEXT: Pass:            loop-vectorize
; REMARKS-NEXT: Name:            MissedDetails
; REMARKS-NEXT: Function:        multiple_predecessors_cond_block
; REMARKS-NEXT: Args:
; REMARKS-NEXT:   - String:          loop not vectorized
; REMARKS-NEXT: ...
define void @multiple_predecessors_cond_block(ptr %ptrA, ptr %ptrB, i1 %cond.load) {
entry:
  indirectbr ptr %ptrA, [label %loop, label %side]

side:
  br label %loop

loop:
  %x = load i64, ptr %ptrB, align 8
  br i1 %cond.load, label %if.then, label %if.end

if.then:
  %y = load i64, ptr %ptrB, align 8
  br label %if.end

if.end:
  br i1 false, label %exit, label %loop

exit:
  ret void
}
