; RUN: opt -passes="scc-oz-module-inliner,function(loop-mssa(no-op-loop)),recompute-globalsaa,function(loop-mssa(simple-loop-unswitch<nontrivial>))" -disable-output < %s
; Check that don't crash if the Alias Analysis returns better results than
; before when cloning loop's memoryssa.

@a = internal global i16 0

define void @h() {
entry:
  br label %end

body:                                       ; No predecessors!
  call void @g(ptr null)
  br label %end

end:                                        ; preds = %while.body, %entry
  ret void
}

define internal void @g(ptr %a) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i16, ptr %a, align 1
  %tobool.not = icmp eq i16 %0, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  call void @f()
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

define internal void @f() #0 {
  store i16 0, ptr @a, align 1
  ret void
}

attributes #0 = { noinline }
