; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes='simple-loop-unswitch<nontrivial>' \
; RUN:     -print-after=simple-loop-unswitch \
; RUN:	   | FileCheck %s

; CHECK: *** IR Dump After SimpleLoopUnswitchPass on for.cond ***
; CHECK: *** IR Dump After SimpleLoopUnswitchPass on for.cond.us ***

define void @loop(i1 %w)  {
entry:
  br label %for.cond
; Loop:
for.cond:                                         ; preds = %for.inc, %entry
  br i1 %w, label %for.inc, label %if.then

if.then:                                          ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.cond
  br label %for.cond
}
