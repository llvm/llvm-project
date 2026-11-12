; RUN: opt -passes="require<globals-aa>,cgscc(instcombine),function(loop-mssa(loop-simplifycfg)),recompute-globalsaa,function(loop-mssa(simple-loop-unswitch<nontrivial>),print<memoryssa>)" -disable-output < %s

; Check that don't crash if the Alias Analysis returns better results than
; before when cloning loop's memoryssa.
define void @f(ptr %p) {
entry:
  %0 = load i16, ptr %p, align 1
  ret void
}

define void @g(i1 %tobool.not) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.then, %for.cond, %entry
  br i1 %tobool.not, label %if.then, label %for.cond

if.then:                                          ; preds = %for.cond
  call void @f()
  br label %for.cond
}
