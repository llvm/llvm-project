; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp -arm-parallel-dsp %s -S -o - | FileCheck %s
; RUN: opt -mtriple=armv6m-none-none-eabi < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=-dsp < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

define i64 @smlaldx(ptr nocapture readonly %pIn1, ptr nocapture readonly %pIn2, i32 %j, i32 %limit) {

; CHECK-LABEL: smlaldx
; CHECK: = phi i32 [ 0, %for.body.preheader.new ],
; CHECK: [[ACC0:%[^ ]+]] = phi i64 [ 0, %for.body.preheader.new ], [ [[ACC2:%[^ ]+]], %for.body ]
; CHECK: [[IN21:%[^ ]+]] = load i32, ptr %pIn2.1, align 2
; CHECK: [[IN10:%[^ ]+]] = load i32, ptr %pIn1.0, align 2
; CHECK: [[ACC1:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[IN21]], i32 [[IN10]], i64 [[ACC0]])
; CHECK: [[IN23:%[^ ]+]] = load i32, ptr %pIn2.3, align 2
; CHECK: [[IN12:%[^ ]+]] = load i32, ptr %pIn1.2, align 2
; CHECK: [[ACC2]] = call i64 @llvm.arm.smlaldx(i32 [[IN23]], i32 [[IN12]], i64 [[ACC1]])
; CHECK-NOT: call i64 @llvm.arm.smlad
; CHECK-UNSUPPORTED-NOT:  call i64 @llvm.arm.smlad

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
  %add.lcssa.ph = phi i64 [ undef, %for.body.preheader ], [ %add.3, %for.body ]
  %i.011.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %sum.010.unr = phi i64 [ 0, %for.body.preheader ], [ %add.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:
  %i.011.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.011.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %sum.010.epil = phi i64 [ %add.epil, %for.body.epil ], [ %sum.010.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %sub.epil = sub i32 %j, %i.011.epil
  %arrayidx.epil = getelementptr inbounds i16, ptr %pIn2, i32 %sub.epil
  %2 = load i16, ptr %arrayidx.epil, align 2
  %conv.epil = sext i16 %2 to i32
  %arrayidx1.epil = getelementptr inbounds i16, ptr %pIn1, i32 %i.011.epil
  %3 = load i16, ptr %arrayidx1.epil, align 2
  %conv2.epil = sext i16 %3 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %sext.mul.epil = sext i32 %mul.epil to i64
  %add.epil = add nsw i64 %sext.mul.epil, %sum.010.epil
  %inc.epil = add nuw i32 %i.011.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:
  %sum.0.lcssa = phi i64 [ 0, %entry ], [ %add.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add.epil, %for.body.epil ]
  ret i64 %sum.0.lcssa

for.body:
  %i.011 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %sum.010 = phi i64 [ 0, %for.body.preheader.new ], [ %add.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %pIn2Base = phi ptr [ %pIn2, %for.body.preheader.new ], [ %pIn2.4, %for.body ]
  %In2 = load i16, ptr %pIn2Base, align 2
  %pIn1.0 = getelementptr inbounds i16, ptr %pIn1, i32 %i.011
  %In1 = load i16, ptr %pIn1.0, align 2
  %inc = or disjoint i32 %i.011, 1
  %pIn2.1 = getelementptr inbounds i16, ptr %pIn2Base, i32 -1
  %In2.1 = load i16, ptr %pIn2.1, align 2
  %pIn1.1 = getelementptr inbounds i16, ptr %pIn1, i32 %inc
  %In1.1 = load i16, ptr %pIn1.1, align 2
  %inc.1 = or disjoint i32 %i.011, 2
  %pIn2.2 = getelementptr inbounds i16, ptr %pIn2Base, i32 -2
  %In2.2 = load i16, ptr %pIn2.2, align 2
  %pIn1.2 = getelementptr inbounds i16, ptr %pIn1, i32 %inc.1
  %In1.2 = load i16, ptr %pIn1.2, align 2
  %inc.2 = or disjoint i32 %i.011, 3
  %pIn2.3 = getelementptr inbounds i16, ptr %pIn2Base, i32 -3
  %In2.3 = load i16, ptr %pIn2.3, align 2
  %pIn1.3 = getelementptr inbounds i16, ptr %pIn1, i32 %inc.2
  %In1.3 = load i16, ptr %pIn1.3, align 2
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
  %sext.mul = sext i32 %mul to i64
  %sext.mul.1 = sext i32 %mul.1 to i64
  %sext.mul.2 = sext i32 %mul.2 to i64
  %sext.mul.3 = sext i32 %mul.3 to i64
  %add = add nsw i64 %sum.010, %sext.mul
  %add.1 = add nsw i64 %sext.mul.1, %add
  %add.2 = add nsw i64 %add.1, %sext.mul.2
  %add.3 = add nsw i64 %sext.mul.3, %add.2
  %inc.3 = add i32 %i.011, 4
  %pIn2.4 = getelementptr inbounds i16, ptr %pIn2Base, i32 -4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

define i64 @smlaldx_swap(ptr nocapture readonly %pIn1, ptr nocapture readonly %pIn2, i32 %j, i32 %limit) {

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
  %scevgep6 = getelementptr i16, ptr %pIn1, i32 2
  %2 = add i32 %j, -1
  %scevgep11 = getelementptr i16, ptr %pIn2, i32 %2
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:
  %add.lcssa.ph = phi i64 [ undef, %for.body.preheader ], [ %add.3, %for.body ]
  %i.011.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %sum.010.unr = phi i64 [ 0, %for.body.preheader ], [ %add.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil.preheader

for.body.epil.preheader:
  %scevgep = getelementptr i16, ptr %pIn1, i32 %i.011.unr
  %3 = sub i32 %j, %i.011.unr
  %scevgep2 = getelementptr i16, ptr %pIn2, i32 %3
  %4 = sub i32 0, %xtraiter
  br label %for.body.epil

for.body.epil:
  %lsr.iv5 = phi i32 [ %4, %for.body.epil.preheader ], [ %lsr.iv.next, %for.body.epil ]
  %lsr.iv3 = phi ptr [ %scevgep2, %for.body.epil.preheader ], [ %scevgep4, %for.body.epil ]
  %lsr.iv = phi ptr [ %scevgep, %for.body.epil.preheader ], [ %scevgep1, %for.body.epil ]
  %sum.010.epil = phi i64 [ %add.epil, %for.body.epil ], [ %sum.010.unr, %for.body.epil.preheader ]
  %5 = load i16, ptr %lsr.iv3, align 2
  %conv.epil = sext i16 %5 to i32
  %6 = load i16, ptr %lsr.iv, align 2
  %conv2.epil = sext i16 %6 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %sext.mul.epil = sext i32 %mul.epil to i64
  %add.epil = add nsw i64 %sext.mul.epil, %sum.010.epil
  %scevgep1 = getelementptr i16, ptr %lsr.iv, i32 1
  %scevgep4 = getelementptr i16, ptr %lsr.iv3, i32 -1
  %lsr.iv.next = add nsw i32 %lsr.iv5, 1
  %epil.iter.cmp = icmp eq i32 %lsr.iv.next, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:
  %sum.0.lcssa = phi i64 [ 0, %entry ], [ %add.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add.epil, %for.body.epil ]
  ret i64 %sum.0.lcssa

; CHECK-LABEL: smlaldx_swap
; CHECK: for.body.preheader.new:
; CHECK: [[PIN1Base:[^ ]+]] = getelementptr i16, ptr %pIn1
; CHECK: [[PIN2Base:[^ ]+]] = getelementptr i16, ptr %pIn2

; CHECK: for.body:
; CHECK: [[PIN2:%[^ ]+]] = phi ptr [ [[PIN2_NEXT:%[^ ]+]], %for.body ], [ [[PIN2Base]], %for.body.preheader.new ]
; CHECK: [[PIN1:%[^ ]+]] = phi ptr [ [[PIN1_NEXT:%[^ ]+]], %for.body ], [ [[PIN1Base]], %for.body.preheader.new ]
; CHECK: [[IV:%[^ ]+]] = phi i32
; CHECK: [[ACC0:%[^ ]+]] = phi i64 [ 0, %for.body.preheader.new ], [ [[ACC2:%[^ ]+]], %for.body ]

; CHECK: [[IN2:%[^ ]+]] = load i32, ptr [[PIN2]], align 2

; CHECK: [[PIN1_2:%[^ ]+]] = getelementptr i16, ptr [[PIN1]], i32 -2
; CHECK: [[IN1_2:%[^ ]+]] = load i32, ptr [[PIN1_2]], align 2
; CHECK: [[ACC1:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[IN2]], i32 [[IN1_2]], i64 [[ACC0]])

; CHECK: [[PIN2_2:%[^ ]+]] = getelementptr i16, ptr [[PIN2]], i32 -2
; CHECK: [[IN2_2:%[^ ]+]] = load i32, ptr [[PIN2_2]], align 2

; CHECK: [[IN1:%[^ ]+]] = load i32, ptr [[PIN1]], align 2
; CHECK: [[ACC2]] = call i64 @llvm.arm.smlaldx(i32 [[IN2_2]], i32 [[IN1]], i64 [[ACC1]])

; CHECK: [[PIN1_NEXT]] = getelementptr i16, ptr [[PIN1]], i32 4
; CHECK: [[PIN2_NEXT]] = getelementptr i16, ptr [[PIN2]], i32 -4

; CHECK-NOT: call i64 @llvm.arm.smlad
; CHECK-UNSUPPORTED-NOT:  call i64 @llvm.arm.smlad

for.body:
  %pin2 = phi ptr [ %pin2.sub4, %for.body ], [ %scevgep11, %for.body.preheader.new ]
  %pin1 = phi ptr [ %pin1.add4, %for.body ], [ %scevgep6, %for.body.preheader.new ]
  %i.011 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %sum.010 = phi i64 [ 0, %for.body.preheader.new ], [ %add.3, %for.body ]
  %pin2.add1 = getelementptr i16, ptr %pin2, i32 1
  %In2 = load i16, ptr %pin2.add1, align 2
  %pin1.sub2 = getelementptr i16, ptr %pin1, i32 -2
  %In1 = load i16, ptr %pin1.sub2, align 2
  %In2.1 = load i16, ptr %pin2, align 2
  %pin1.sub1 = getelementptr i16, ptr %pin1, i32 -1
  %In1.1 = load i16, ptr %pin1.sub1, align 2
  %pin2.sub1 = getelementptr i16, ptr %pin2, i32 -1
  %In2.2 = load i16, ptr %pin2.sub1, align 2
  %In1.2 = load i16, ptr %pin1, align 2
  %pin2.sub2 = getelementptr i16, ptr %pin2, i32 -2
  %In2.3 = load i16, ptr %pin2.sub2, align 2
  %pin1.add1 = getelementptr i16, ptr %pin1, i32 1
  %In1.3 = load i16, ptr %pin1.add1, align 2
  %sextIn2 = sext i16 %In2 to i32
  %sextIn1 = sext i16 %In1 to i32
  %sextIn2.1 = sext i16 %In2.1 to i32
  %sextIn1.1 = sext i16 %In1.1 to i32
  %sextIn2.2 = sext i16 %In2.2 to i32
  %sextIn1.2 = sext i16 %In1.2 to i32
  %sextIn2.3 = sext i16 %In2.3 to i32
  %sextIn1.3 = sext i16 %In1.3 to i32
  %mul = mul nsw i32 %sextIn2, %sextIn1
  %sext.mul = sext i32 %mul to i64
  %add = add nsw i64 %sext.mul, %sum.010
  %mul.1 = mul nsw i32 %sextIn2.1, %sextIn1.1
  %sext.mul.1 = sext i32 %mul.1 to i64
  %add.1 = add nsw i64 %sext.mul.1, %add
  %mul.2 = mul nsw i32 %sextIn2.2, %sextIn1.2
  %sext.mul.2 = sext i32 %mul.2 to i64
  %add.2 = add nsw i64 %add.1, %sext.mul.2
  %mul.3 = mul nsw i32 %sextIn2.3, %sextIn1.3
  %sext.mul.3 = sext i32 %mul.3 to i64
  %add.3 = add nsw i64 %add.2, %sext.mul.3
  %inc.3 = add i32 %i.011, 4
  %pin1.add4 = getelementptr i16, ptr %pin1, i32 4
  %pin2.sub4 = getelementptr i16, ptr %pin2, i32 -4
  %niter.ncmp.3 = icmp eq i32 %unroll_iter, %inc.3
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}
