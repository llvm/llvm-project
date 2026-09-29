; RUN: opt -passes='default<O3>' -enable-loopinterchange -disable-output \
; RUN:     -disable-verify -verify-analysis-invalidation=0 \
; RUN:     -debug-pass-manager=quiet %s 2>&1 | FileCheck %s

; Test the position of LoopInterchange in the pass pipeline.

; CHECK-NOT:  Running pass: LoopInterchangePass
; CHECK:      Running pass: ControlHeightReductionPass
; CHECK-NEXT: Running pass: LoopSimplifyPass
; CHECK-NEXT: Running pass: LCSSAPass
; CHECK-NEXT: Running pass: LoopRotatePass
; CHECK-NEXT: Running pass: LoopDeletionPass
; CHECK-NEXT: Running pass: LoopRotatePass
; CHECK-NEXT: Running pass: LoopDeletionPass
; CHECK-NEXT: Running pass: LoopInterchangePass
; CHECK-NEXT: Running pass: LoopDistributePass
; CHECK-NEXT: Running pass: InjectTLIMappings
; CHECK-NEXT: Running pass: LoopVectorizePass


define void @foo(ptr %a, i32 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.next, %for.j ]
  %tmp = mul i32 %i, %n
  %offset = add i32 %tmp, %j
  %idx = getelementptr inbounds i32, ptr %a, i32 %offset
  %load = load i32, ptr %idx, align 4
  %inc = add i32 %load, 1
  store i32 %inc, ptr %idx, align 4
  %j.next = add i32 %j, 1
  %j.exit = icmp eq i32 %j.next, %n
  br i1 %j.exit, label %for.i.latch, label %for.j

for.i.latch:
  %i.next = add i32 %i, 1
  %i.exit = icmp eq i32 %i.next, %n
  br i1 %i.exit, label %for.i.header, label %exit

exit:
  ret void
}
