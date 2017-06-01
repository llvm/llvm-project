; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the HexagonExpandCondsets pass does not assert due to
; attempting to shrink a live interval incorrectly.


define void @test() #0 {
entry:
  br i1 undef, label %cleanup, label %if.end

if.end:
  %0 = load i32, i32* undef, align 4
  %sext = shl i32 %0, 16
  %conv19 = ashr exact i32 %sext, 16
  br i1 undef, label %cleanup, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.body:
  %bestScoreL16Q4.0278 = phi i16 [ 32767, %for.body.lr.ph ], [ %.sink, %early_termination ]
  br i1 false, label %for.body44.lr.ph, label %for.cond90.preheader

for.body44.lr.ph:
  %conv77 = sext i16 %bestScoreL16Q4.0278 to i32
  unreachable

for.cond90.preheader:
  br i1 undef, label %early_termination, label %for.body97

for.body97:
  br i1 undef, label %for.body97, label %early_termination

early_termination:
  %.sink = select i1 undef, i16 undef, i16 %bestScoreL16Q4.0278
  %cmp27 = icmp slt i32 undef, %conv19
  br i1 %cmp27, label %for.body, label %for.end124

for.end124:
  unreachable

cleanup:
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
