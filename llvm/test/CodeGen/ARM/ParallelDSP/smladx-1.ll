; RUN: opt -mtriple=thumbv8m.main -mcpu=cortex-m33 -arm-parallel-dsp %s -S -o - | FileCheck %s
; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m0 < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m33 -mattr=-dsp < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
; RUN: opt -mtriple=armeb-arm-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

define i32 @smladx(i16* nocapture readonly %pIn1, i16* nocapture readonly %pIn2, i32 %j, i32 %limit) {

; CHECK-LABEL: smladx
; CHECK: = phi i32 [ 0, %for.body.preheader.new ],
; CHECK: [[ACC0:%[^ ]+]] = phi i32 [ 0, %for.body.preheader.new ], [ [[ACC2:%[^ ]+]], %for.body ]
; CHECK: [[PIN21:%[^ ]+]] = bitcast i16* %pIn2.1 to i32*
; CHECK: [[IN21:%[^ ]+]] = load i32, i32* [[PIN21]], align 2
; CHECK: [[PIN10:%[^ ]+]] = bitcast i16* %pIn1.0 to i32*
; CHECK: [[IN10:%[^ ]+]] = load i32, i32* [[PIN10]], align 2
; CHECK: [[ACC1:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN21]], i32 [[IN10]], i32 [[ACC0]])

; CHECK: [[PIN23:%[^ ]+]] = bitcast i16* %pIn2.3 to i32*
; CHECK: [[IN23:%[^ ]+]] = load i32, i32* [[PIN23]], align 2
; CHECK: [[PIN12:%[^ ]+]] = bitcast i16* %pIn1.2 to i32*
; CHECK: [[IN12:%[^ ]+]] = load i32, i32* [[PIN12]], align 2
; CHECK: [[ACC2]] = call i32 @llvm.arm.smladx(i32 [[IN23]], i32 [[IN12]], i32 [[ACC1]])
; CHECK-NOT: call i32 @llvm.arm.smlad
; CHECK-UNSUPPORTED-NOT:  call i32 @llvm.arm.smlad

entry:
  %cmp9 = icmp eq i32 %limit, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  %0 = add i32 %limit, -1
  %xtraiter = and i32 %limit, 3
  %1 = icmp ult i32 %0, 3
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:
  %unroll_iter = sub i32 %limit, %xtraiter
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:
  %add.lcssa.ph = phi i32 [ undef, %for.body.preheader ], [ %add.3, %for.body ]
  %i.011.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %sum.010.unr = phi i32 [ 0, %for.body.preheader ], [ %add.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:
  %i.011.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.011.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %sum.010.epil = phi i32 [ %add.epil, %for.body.epil ], [ %sum.010.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %sub.epil = sub i32 %j, %i.011.epil
  %arrayidx.epil = getelementptr inbounds i16, i16* %pIn2, i32 %sub.epil
  %2 = load i16, i16* %arrayidx.epil, align 2
  %conv.epil = sext i16 %2 to i32
  %arrayidx1.epil = getelementptr inbounds i16, i16* %pIn1, i32 %i.011.epil
  %3 = load i16, i16* %arrayidx1.epil, align 2
  %conv2.epil = sext i16 %3 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %add.epil = add nsw i32 %mul.epil, %sum.010.epil
  %inc.epil = add nuw i32 %i.011.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add.epil, %for.body.epil ]
  ret i32 %sum.0.lcssa

for.body:
  %i.011 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %sum.010 = phi i32 [ 0, %for.body.preheader.new ], [ %add.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %pIn2Base = phi i16* [ %pIn2, %for.body.preheader.new ], [ %pIn2.4, %for.body ]
  %pIn2.0 = getelementptr inbounds i16, i16* %pIn2Base, i32 0
  %In2 = load i16, i16* %pIn2.0, align 2
  %pIn1.0 = getelementptr inbounds i16, i16* %pIn1, i32 %i.011
  %In1 = load i16, i16* %pIn1.0, align 2
  %inc = or i32 %i.011, 1
  %pIn2.1 = getelementptr inbounds i16, i16* %pIn2Base, i32 -1
  %In2.1 = load i16, i16* %pIn2.1, align 2
  %pIn1.1 = getelementptr inbounds i16, i16* %pIn1, i32 %inc
  %In1.1 = load i16, i16* %pIn1.1, align 2
  %inc.1 = or i32 %i.011, 2
  %pIn2.2 = getelementptr inbounds i16, i16* %pIn2Base, i32 -2
  %In2.2 = load i16, i16* %pIn2.2, align 2
  %pIn1.2 = getelementptr inbounds i16, i16* %pIn1, i32 %inc.1
  %In1.2 = load i16, i16* %pIn1.2, align 2
  %inc.2 = or i32 %i.011, 3
  %pIn2.3 = getelementptr inbounds i16, i16* %pIn2Base, i32 -3
  %In2.3 = load i16, i16* %pIn2.3, align 2
  %pIn1.3 = getelementptr inbounds i16, i16* %pIn1, i32 %inc.2
  %In1.3 = load i16, i16* %pIn1.3, align 2
  %sextIn1 = sext i16 %In1 to i32
  %sextIn1.1 = sext i16 %In1.1 to i32
  %sextIn1.2 = sext i16 %In1.2 to i32
  %sextIn1.3 = sext i16 %In1.3 to i32
  %sextIn2 = sext i16 %In2 to i32
  %sextIn2.1 = sext i16 %In2.1 to i32
  %sextIn2.2 = sext i16 %In2.2 to i32
  %sextIn2.3 = sext i16 %In2.3 to i32
  %mul = mul nsw i32 %sextIn1, %sextIn2
  %mul.1 = mul nsw i32 %sextIn1.1, %sextIn2.1
  %mul.2 = mul nsw i32 %sextIn1.2, %sextIn2.2
  %mul.3 = mul nsw i32 %sextIn1.3, %sextIn2.3
  %add = add nsw i32 %mul, %sum.010
  %add.1 = add nsw i32 %mul.1, %add
  %add.2 = add nsw i32 %mul.2, %add.1
  %add.3 = add nsw i32 %mul.3, %add.2
  %inc.3 = add i32 %i.011, 4
  %pIn2.4 = getelementptr inbounds i16, i16* %pIn2Base, i32 -4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

define i32 @smladx_swap(i16* nocapture readonly %pIn1, i16* nocapture readonly %pIn2, i32 %j, i32 %limit) {

; CHECK-LABEL: smladx_swap
; CHECK: for.body.preheader.new:
; CHECK: [[PIN1Base:[^ ]+]] = getelementptr i16, i16* %pIn1
; CHECK: [[PIN2Base:[^ ]+]] = getelementptr i16, i16* %pIn2

; CHECK: for.body:
; CHECK: [[PIN2:%[^ ]+]] = phi i16* [ [[PIN2_NEXT:%[^ ]+]], %for.body ], [ [[PIN2Base]], %for.body.preheader.new ]
; CHECK: [[PIN1:%[^ ]+]] = phi i16* [ [[PIN1_NEXT:%[^ ]+]], %for.body ], [ [[PIN1Base]], %for.body.preheader.new ]
; CHECK: [[IV:%[^ ]+]] = phi i32
; CHECK: [[ACC0:%[^ ]+]] = phi i32 [ 0, %for.body.preheader.new ], [ [[ACC2:%[^ ]+]], %for.body ]

; CHECK: [[PIN2_CAST:%[^ ]+]] = bitcast i16* [[PIN2]] to i32*
; CHECK: [[IN2:%[^ ]+]] = load i32, i32* [[PIN2_CAST]], align 2

; CHECK: [[PIN1_2:%[^ ]+]] = getelementptr i16, i16* [[PIN1]], i32 -2
; CHECK: [[PIN1_2_CAST:%[^ ]+]] = bitcast i16* [[PIN1_2]] to i32*
; CHECK: [[IN1_2:%[^ ]+]] = load i32, i32* [[PIN1_2_CAST]], align 2
; CHECK: [[ACC1:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN2]], i32 [[IN1_2]], i32 [[ACC0]])

; CHECK: [[PIN2_2:%[^ ]+]] = getelementptr i16, i16* [[PIN2]], i32 -2
; CHECK: [[PIN2_2_CAST:%[^ ]+]] = bitcast i16* [[PIN2_2]] to i32*
; CHECK: [[IN2_2:%[^ ]+]] = load i32, i32* [[PIN2_2_CAST]], align 2

; CHECK: [[PIN1_CAST:%[^ ]+]] = bitcast i16* [[PIN1]] to i32*
; CHECK: [[IN1:%[^ ]+]] = load i32, i32* [[PIN1_CAST]], align 2

; CHECK: [[ACC2]] = call i32 @llvm.arm.smladx(i32 [[IN2_2]], i32 [[IN1]], i32 [[ACC1]])

; CHECK: [[PIN1_NEXT]] = getelementptr i16, i16* [[PIN1]], i32 4
; CHECK: [[PIN2_NEXT]] = getelementptr i16, i16* [[PIN2]], i32 -4

; CHECK-NOT: call i32 @llvm.arm.smlad
; CHECK-UNSUPPORTED-NOT:  call i32 @llvm.arm.smlad

entry:
  %cmp9 = icmp eq i32 %limit, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  %0 = add i32 %limit, -1
  %xtraiter = and i32 %limit, 3
  %1 = icmp ult i32 %0, 3
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:
  %unroll_iter = sub i32 %limit, %xtraiter
  %scevgep6 = getelementptr i16, i16* %pIn1, i32 2
  %2 = add i32 %j, -1
  %scevgep11 = getelementptr i16, i16* %pIn2, i32 %2
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:
  %add.lcssa.ph = phi i32 [ undef, %for.body.preheader ], [ %add.3, %for.body ]
  %i.011.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %sum.010.unr = phi i32 [ 0, %for.body.preheader ], [ %add.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil.preheader

for.body.epil.preheader:
  %scevgep = getelementptr i16, i16* %pIn1, i32 %i.011.unr
  %3 = sub i32 %j, %i.011.unr
  %scevgep2 = getelementptr i16, i16* %pIn2, i32 %3
  %4 = sub i32 0, %xtraiter
  br label %for.body.epil

for.body.epil:
  %lsr.iv5 = phi i32 [ %4, %for.body.epil.preheader ], [ %lsr.iv.next, %for.body.epil ]
  %lsr.iv3 = phi i16* [ %scevgep2, %for.body.epil.preheader ], [ %scevgep4, %for.body.epil ]
  %lsr.iv = phi i16* [ %scevgep, %for.body.epil.preheader ], [ %scevgep1, %for.body.epil ]
  %sum.010.epil = phi i32 [ %add.epil, %for.body.epil ], [ %sum.010.unr, %for.body.epil.preheader ]
  %5 = load i16, i16* %lsr.iv3, align 2
  %conv.epil = sext i16 %5 to i32
  %6 = load i16, i16* %lsr.iv, align 2
  %conv2.epil = sext i16 %6 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %add.epil = add nsw i32 %mul.epil, %sum.010.epil
  %scevgep1 = getelementptr i16, i16* %lsr.iv, i32 1
  %scevgep4 = getelementptr i16, i16* %lsr.iv3, i32 -1
  %lsr.iv.next = add nsw i32 %lsr.iv5, 1
  %epil.iter.cmp = icmp eq i32 %lsr.iv.next, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add.epil, %for.body.epil ]
  ret i32 %sum.0.lcssa

for.body:
  %pin2 = phi i16* [ %pin2_sub4, %for.body ], [ %scevgep11, %for.body.preheader.new ]
  %pin1 = phi i16* [ %pin1_add4, %for.body ], [ %scevgep6, %for.body.preheader.new ]
  %i.011 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %sum.010 = phi i32 [ 0, %for.body.preheader.new ], [ %add.3, %for.body ]
  %pin2_add1 = getelementptr i16, i16* %pin2, i32 1
  %In2 = load i16, i16* %pin2_add1, align 2
  %pin1_sub2 = getelementptr i16, i16* %pin1, i32 -2
  %In1 = load i16, i16* %pin1_sub2, align 2
  %In2.1 = load i16, i16* %pin2, align 2
  %pin1_sub1 = getelementptr i16, i16* %pin1, i32 -1
  %In1.1 = load i16, i16* %pin1_sub1, align 2
  %pin2_sub1 = getelementptr i16, i16* %pin2, i32 -1
  %In2.2 = load i16, i16* %pin2_sub1, align 2
  %In1.2 = load i16, i16* %pin1, align 2
  %pin2_sub2 = getelementptr i16, i16* %pin2, i32 -2
  %In2.3 = load i16, i16* %pin2_sub2, align 2
  %pin1_add1 = getelementptr i16, i16* %pin1, i32 1
  %In1.3 = load i16, i16* %pin1_add1, align 2
  %sextIn2 = sext i16 %In2 to i32
  %sextIn1 = sext i16 %In1 to i32
  %sextIn2.1 = sext i16 %In2.1 to i32
  %sextIn1.1 = sext i16 %In1.1 to i32
  %sextIn2.2 = sext i16 %In2.2 to i32
  %sextIn1.2 = sext i16 %In1.2 to i32
  %sextIn2.3 = sext i16 %In2.3 to i32
  %sextIn1.3 = sext i16 %In1.3 to i32
  %mul = mul nsw i32 %sextIn2, %sextIn1
  %add = add nsw i32 %mul, %sum.010
  %mul.1 = mul nsw i32 %sextIn2.1, %sextIn1.1
  %add.1 = add nsw i32 %mul.1, %add
  %mul.2 = mul nsw i32 %sextIn2.2, %sextIn1.2
  %add.2 = add nsw i32 %mul.2, %add.1
  %mul.3 = mul nsw i32 %sextIn2.3, %sextIn1.3
  %add.3 = add nsw i32 %mul.3, %add.2
  %inc.3 = add i32 %i.011, 4
  %pin1_add4 = getelementptr i16, i16* %pin1, i32 4
  %pin2_sub4 = getelementptr i16, i16* %pin2, i32 -4
  %niter.ncmp.3 = icmp eq i32 %unroll_iter, %inc.3
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}
