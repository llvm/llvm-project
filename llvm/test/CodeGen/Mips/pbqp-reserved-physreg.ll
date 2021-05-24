; RUN: llc -march=mips -regalloc=pbqp <%s > %t
; ModuleID = 'bugpoint-reduced-simplified.bc'

; Function Attrs: nounwind
define void @ham.928() local_unnamed_addr #0 align 2 {
bb:
  switch i32 undef, label %bb35 [
    i32 1, label %bb18
    i32 0, label %bb19
    i32 3, label %bb20
    i32 2, label %bb21
    i32 4, label %bb17
  ]

bb17:                                             ; preds = %bb
  unreachable

bb18:                                             ; preds = %bb
  unreachable

bb19:                                             ; preds = %bb
  unreachable

bb20:                                             ; preds = %bb
  unreachable

bb21:                                             ; preds = %bb
  unreachable

bb35:                                             ; preds = %bb
  unreachable
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "stack-protector-buffer-size"="8" "target-cpu"="generic" "use-soft-float"="false" }

