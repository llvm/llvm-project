; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; The two loads based on %struct.0, loading two different data types
; cause LSR to assume type "void" for the memory type. This would then
; cause an assert in isLegalAddressingMode. Make sure we no longer crash.

target triple = "hexagon"

%struct.0 = type { ptr, i8, %union.anon.0 }
%union.anon.0 = type { ptr }

define hidden fastcc void @fred() unnamed_addr #0 {
entry:
  br i1 undef, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %exit.2, %while.body.lr.ph
  %lsr.iv = phi ptr [ %cgep22, %exit.2 ], [ undef, %while.body.lr.ph ]
  switch i32 undef, label %exit [
    i32 1, label %sw.bb.i
    i32 2, label %sw.bb3.i
  ]

sw.bb.i:                                          ; preds = %while.body
  unreachable

sw.bb3.i:                                         ; preds = %while.body
  unreachable

exit:                                             ; preds = %while.body
  switch i32 undef, label %exit.2 [
    i32 1, label %sw.bb.i17
    i32 2, label %sw.bb3.i20
  ]

sw.bb.i17:                                        ; preds = %.exit
  %0 = load i32, ptr %lsr.iv, align 4
  unreachable

sw.bb3.i20:                                       ; preds = %exit
  %1 = load ptr, ptr %lsr.iv, align 4
  unreachable

exit.2:                                           ; preds = %exit
  %cgep22 = getelementptr %struct.0, ptr %lsr.iv, i32 1
  br label %while.body

while.end:                                        ; preds = %entry
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" }

