; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; This test used to crash in register scavenger due to incorrectly set
; kill flags.

target triple = "hexagon"

define void @foo(i64 %a) #0 {
entry:
  %conv.i = and i64 %a, 9218868437227405312
  %cmp = icmp ne i64 %conv.i, 9218868437227405312
  %and.i37 = and i64 %a, 4503599627370495
  %tobool = icmp eq i64 %and.i37, 0
  %or.cond = or i1 %cmp, %tobool
  br i1 %or.cond, label %lor.lhs.false, label %if.then

lor.lhs.false:                                    ; preds = %entry
  br i1 undef, label %return, label %if.then

if.then:                                          ; preds = %lor.lhs.false, %entry
  br label %return

return:                                           ; preds = %if.then, %lor.lhs.false
  ret void
}

attributes #0 = { norecurse nounwind }
