; RUN: llc -O3 -mtriple=arm-arm-eabi -mcpu=cortex-m33 < %s | FileCheck %s
; RUN: llc -O3 -mtriple=armeb-arm-eabi -mcpu=cortex-m33 < %s | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

; CHECK-UNSUPPORTED-NOT: smlad

; CHECK-LABEL: add_user
; CHECK: %for.body
; CHECK: ldr [[A:[rl0-9]+]],{{.*}}, #2]!
; CHECK: ldr [[B:[rl0-9]+]],{{.*}}, #2]!
; CHECK: sxtah [[COUNT:r[0-9]+]], [[COUNT]], [[A]]
; CHECK: smlad [[ACC:r[0-9]+]], [[B]], [[A]], [[ACC]]
define i32 @add_user(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %count.final = phi i32 [ 0, %entry ], [ %count.next, %for.body ]
  %res = add i32 %mac1.0.lcssa, %count.final
  ret i32 %res

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %count = phi i32 [ %count.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %count.next = add i32 %conv4, %count
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: mul_bottom_user
; CHECK: %for.body
; CHECK: ldr [[A:[rl0-9]+]],{{.*}}, #2]!
; CHECK: ldr [[B:[rl0-9]+]],{{.*}}, #2]!
; CHECK: sxth [[SXT:r[0-9]+]], [[A]]
; CHECK: smlad [[ACC:r[0-9]+]], [[B]], [[A]], [[ACC]]
; CHECK: mul [[COUNT:r[0-9]+]],{{.*}}[[SXT]]
define i32 @mul_bottom_user(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %count.final = phi i32 [ 0, %entry ], [ %count.next, %for.body ]
  %res = add i32 %mac1.0.lcssa, %count.final
  ret i32 %res

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %count = phi i32 [ %count.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %count.next = mul i32 %conv4, %count
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: mul_top_user
; CHECK: %for.body
; CHECK: ldr [[A:[rl0-9]+]],{{.*}}, #2]!
; CHECK: ldr [[B:[rl0-9]+]],{{.*}}, #2]!
; CHECK: asrs [[ASR:[rl0-9]+]], [[A]], #16
; CHECK: smlad [[ACC:[rl0-9]+]], [[A]], [[B]], [[ACC]]
; CHECK: mul [[COUNT:[rl0-9]+]],{{.}}[[ASR]]
define i32 @mul_top_user(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %count.final = phi i32 [ 0, %entry ], [ %count.next, %for.body ]
  %res = add i32 %mac1.0.lcssa, %count.final
  ret i32 %res

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %count = phi i32 [ %count.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %count.next = mul i32 %conv7, %count
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: and_user
; CHECK: %for.body
; CHECK: ldr [[A:[rl0-9]+]],{{.*}}, #2]!
; CHECK: ldr [[B:[rl0-9]+]],{{.*}}, #2]!
; CHECK: uxth [[UXT:r[0-9]+]], [[A]]
; CHECK: smlad [[ACC:r[0-9]+]], [[B]], [[A]], [[ACC]]
; CHECK: mul [[MUL:r[0-9]+]],{{.*}}[[UXT]]
define i32 @and_user(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %count.final = phi i32 [ 0, %entry ], [ %count.next, %for.body ]
  %res = add i32 %mac1.0.lcssa, %count.final
  ret i32 %res

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %count = phi i32 [ %count.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %bottom = and i32 %conv4, 65535
  %mul = mul nsw i32 %conv, %conv4
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %count.next = mul i32 %bottom, %count
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: multi_uses
; CHECK: %for.body
; CHECK: ldr [[A:[rl0-9]+]], [{{.*}}, #2]!
; CHECK: ldr [[B:[rl0-9]+]], [{{.*}}, #2]!
; CHECK: sxth [[SXT:r[0-9]+]], [[A]]
; CHECK: smlad [[ACC:[rl0-9]+]], [[B]], [[A]], [[ACC]]
; CHECK: eor.w [[EOR:r[0-9]+]], [[SXT]], [[SHIFT:r[0-9]+]]
; CHECK: muls [[MUL:r[0-9]+]],{{.*}}[[SXT]]
; CHECK: lsl.w [[SHIFT]], [[MUL]], #16
define i32 @multi_uses(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %count.final = phi i32 [ 0, %entry ], [ %count.next, %for.body ]
  %res = add i32 %mac1.0.lcssa, %count.final
  ret i32 %res

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %count = phi i32 [ %count.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %bottom = and i32 %conv4, 65535
  %mul = mul nsw i32 %conv, %conv4
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %shl = shl i32 %conv4, 16
  %add11 = add i32 %mul9, %add10
  %xor = xor i32 %bottom, %count
  %count.next = mul i32 %xor, %shl
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}
