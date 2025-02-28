; RUN: llc < %s -mtriple=arm-eabi -mcpu=generic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6m-eabi | FileCheck %s -check-prefix=CHECK-V6M-THUMB

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
; CHECK-V6M-THUMB-LABEL: smul:
; CHECK-V6M-THUMB: bl __aeabi_lmul
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
; CHECK-V6M-THUMB-LABEL: umul:
; CHECK-V6M-THUMB: bl __aeabi_lmul
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

define void @sum(ptr %a, ptr %b, i32 %n) local_unnamed_addr #0 {
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
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i32 %i.08
  %1 = load i32, ptr %arrayidx1, align 4
  %2 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %1, i32 %0)
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %trap, label %cont

trap:
  tail call void @llvm.trap() #2
  unreachable

cont:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr %arrayidx1, align 4
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
; CHECK: bl external_fn
; CHECK-NEXT: adds
; CHECK-NEXT: bvs
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
  tail call void @external_fn() #4
  %3 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.046, i32 1)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont1

cont1:
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp = icmp slt i32 %5, %2
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare void @external_fn(...) local_unnamed_addr #0

define i32 @are_equal(ptr nocapture readonly %a1, ptr nocapture readonly %a2, i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: are_equal
; CHECK: subs r{{[0-9]+}}, r{{[0-9]+}}, #1
; CHECK-NEXT: bne
entry:
  %tobool7 = icmp eq i32 %n, 0
  br i1 %tobool7, label %while.end, label %land.rhs.preheader

land.rhs.preheader:
  br label %land.rhs

while.cond:
  %tobool = icmp eq i32 %dec9, 0
  br i1 %tobool, label %while.end, label %land.rhs

land.rhs:
  %dec9.in = phi i32 [ %dec9, %while.cond ], [ %n, %land.rhs.preheader ]
  %dec9 = add nsw i32 %dec9.in, -1
  %arrayidx = getelementptr inbounds i32, ptr %a1, i32 %dec9
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %a2, i32 %dec9
  %1 = load i32, ptr %arrayidx1, align 4
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %while.cond, label %while.end

while.end:
  %n.addr.0.lcssa = phi i32 [ 0, %entry ], [ 0, %while.cond ], [ %dec9.in, %land.rhs ]
  %cmp2 = icmp slt i32 %n.addr.0.lcssa, 1
  %conv = zext i1 %cmp2 to i32
  ret i32 %conv
}

define i1 @no__mulodi4(i32 %a, i64 %b, ptr %c) {
; CHECK-LABEL: no__mulodi4
; CHECK-NOT: bl __mulodi4
entry:
  %0 = sext i32 %a to i64
  %1 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %0, i64 %b)
  %2 = extractvalue { i64, i1 } %1, 1
  %3 = extractvalue { i64, i1 } %1, 0
  %4 = trunc i64 %3 to i32
  %5 = sext i32 %4 to i64
  %6 = icmp ne i64 %3, %5
  %7 = or i1 %2, %6
  store i32 %4, ptr %c, align 4
  ret i1 %7
}

declare void @llvm.trap() #2
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #1
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)
