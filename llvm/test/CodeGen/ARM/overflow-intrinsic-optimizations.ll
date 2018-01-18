; RUN: llc < %s -mtriple=arm-eabi -mcpu=generic | FileCheck %s

define i32 @sadd(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: sadd:
; CHECK:    adds r0, r0, r1
; CHECK-NEXT:    movvc pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @uadd(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: uadd:
; CHECK:    adds r0, r0, r1
; CHECK-NEXT:    movlo pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @ssub(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: ssub:
; CHECK:    subs r0, r0, r1
; CHECK-NEXT:    movvc pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @usub(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: usub:
; CHECK:    subs r0, r0, r1
; CHECK-NEXT:    movhs pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2

}

define i32 @smul(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: smul:
; CHECK: smull r0, r[[RHI:[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}
; CHECK-NEXT: cmp r[[RHI]], r0, asr #31
; CHECK-NEXT: moveq pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2
}

define i32 @umul(i32 %a, i32 %b) local_unnamed_addr #0 {
; CHECK-LABEL: umul:
; CHECK: umull r0, r[[RHI:[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}
; CHECK-NEXT: cmp r[[RHI]], #0
; CHECK-NEXT: moveq pc, lr
entry:
  %0 = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %2 = extractvalue { i32, i1 } %0, 0
  ret i32 %2
}

define void @sum(i32* %a, i32* %b, i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: sum:
; CHECK:    ldr [[R0:r[0-9]+]],
; CHECK-NEXT:    ldr [[R1:r[0-9]+|lr]],
; CHECK-NEXT:    adds [[R2:r[0-9]+]], [[R1]], [[R0]]
; CHECK-NEXT:    strvc [[R2]],
; CHECK-NEXT:    addsvc
; CHECK-NEXT:    bvs
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.08 = phi i32 [ %7, %cont2 ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i32 %i.08
  %1 = load i32, i32* %arrayidx1, align 4
  %2 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %1, i32 %0)
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, i32* %arrayidx1, align 4
  %5 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.08, i32 1)
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %trap, label %cont2

cont2:
  %7 = extractvalue { i32, i1 } %5, 0
  %cmp = icmp eq i32 %7, %n
  br i1 %cmp, label %for.cond.cleanup, label %for.body

}

define void @extern_loop(i32 %n) local_unnamed_addr #0 {
; Do not replace the compare around the clobbering call.
; CHECK: add {{r[0-9]+}}, {{r[0-9]+}}, #1
; CHECK-NEXT: bl external_fn
; CHECK: cmp
entry:
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %n, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont.lr.ph

cont.lr.ph:
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp5 = icmp sgt i32 %2, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

trap:
  tail call void @llvm.trap() #2
  unreachable

for.cond.cleanup:
  ret void

for.body:
  %i.046 = phi i32 [ %5, %cont1 ], [ 0, %for.body.preheader ]
  tail call void bitcast (void (...)* @external_fn to void ()*)() #4
  %3 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.046, i32 1)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont1

cont1:
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp = icmp slt i32 %5, %2
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare void @external_fn(...) local_unnamed_addr #0

declare void @llvm.trap() #2
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #1
