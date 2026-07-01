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
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ], [ %iv.next, %bb ]
  %iv.next = add nuw nsw i64 %iv, 1
  %self.latch = icmp ult i64 %iv.next, 16
  br i1 %self.latch, label %loop, label %bb

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
  %iv = phi i64 [ 0, %entry ], [ 0, %side ], [ %iv.next, %loop ], [ %iv.next, %back ]
  %x = load i64, ptr %ptrB, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %self.latch = icmp ult i64 %iv.next, 16
  br i1 %self.latch, label %loop, label %back

back:
  br label %loop
}

; REMARKS:      Name:            CFGNotUnderstood
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
  %iv = phi i64 [ 0, %entry ], [ 0, %side ], [ %iv.next, %if.end ]
  %x = load i64, ptr %ptrB, align 8
  br i1 %cond.load, label %if.then, label %if.end

if.then:
  %y = load i64, ptr %ptrB, align 8
  br label %if.end

if.end:
  %iv.next = add nuw nsw i64 %iv, 1
  %loop.next = icmp ult i64 %iv.next, 32
  br i1 %loop.next, label %loop, label %exit

exit:
  ret void
}
