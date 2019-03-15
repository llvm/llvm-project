; RUN: llc --mtriple=thumbv7em -mattr=+fp-armv8 -O3 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEFAULT --check-prefix=CHECK-T2
; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp -O3 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEFAULT --check-prefix=CHECK-T2
; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp -lsr-backedge-indexing=false %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED
; RUN: llc -mtriple=thumbv8m.base %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED
; RUN: llc -mtriple=thumbv8 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLED
; RUN: llc -mtriple=thumbv8m.main -mattr=+fp-armv8,+dsp -O3 -lsr-complexity-limit=2147483647 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-COMPLEX --check-prefix=CHECK-T2

; Tests to check that post increment addressing modes are used instead of
; updating base pointers with add instructions.

; TODO: I think we should be able to use post inc addressing with VLDM
; instructions.
; CHECK-LABEL: test_fma
; CHECK: @ %loop

; CHECK-DEFAULT: vldr s{{.*}}, #8]
; CHECK-DEFAULT: vldr s{{.*}}, #8]
; CHECK-DEFAULT: vldr s{{.*}}, #12]
; CHECK-DEFAULT: vldr s{{.*}}, #12]

; CHECK-COMPLEX: vldr s{{.*}}, #8]
; CHECK-COMPLEX: vldr s{{.*}}, #8]
; CHECK-COMPLEX: vldr s{{.*}}, #12]
; CHECK-COMPLEX: vldr s{{.*}}, #12]

define float @test_fma(float* %a, float* %b, i32 %N) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx.1 = phi i32 [ 0, %entry ], [ %idx.next, %loop ]
  %res = phi float [ 0.0, %entry ], [ %fma.2, %loop ]
  %gep.a.1 = getelementptr inbounds float, float* %a, i32 %idx.1
  %a.1 = load float, float* %gep.a.1
  %gep.b.1 = getelementptr inbounds float, float* %b, i32 %idx.1
  %b.1 = load float, float* %gep.b.1
  %fmul.1 = fmul float %a.1, %b.1
  %fma.1 = fadd float %fmul.1, %res
  %idx.2 = or i32 %idx.1, 1
  %gep.a.2 = getelementptr inbounds float, float* %a, i32 %idx.2
  %a.2 = load float, float* %gep.a.2
  %gep.b.2 = getelementptr inbounds float, float* %b, i32 %idx.2
  %b.2 = load float, float* %gep.b.2
  %fmul.2 = fmul float %a.2, %b.2
  %fma.2 = fadd float %fmul.2, %fma.1
  %i.next = add nsw nuw i32 %i, -2
  %idx.next = add nsw nuw i32 %idx.1, 2
  %cmp = icmp ult i32 %i.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret float %fma.2
}

; CHECK-LABEL: convolve_16bit
; TODO: Both arrays should use indexing
; CHECK-DEFAULT: ldr{{.*}}, #8]!
; CHECK-DEFAULT-NOT: ldr{{.*}}]!

; CHECK-COMPLEX: ldr{{.*}}, #8]!
; CHECK-COMPLEX-NOT: ldr{{.*}}]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

define void @convolve_16bit(i16** nocapture readonly %input_image, i16** nocapture readonly %filter,
                            i32 %filter_dim, i32 %out_width, i32 %out_height,
                            i32** nocapture readonly %convolved) {
entry:
  %cmp92 = icmp eq i32 %out_height, 0
  br i1 %cmp92, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %xtraiter = and i32 %filter_dim, 3
  %unroll_iter = sub i32 %filter_dim, %xtraiter
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %for.cond1.preheader.lr.ph
  %res_y.093 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %add28, %for.cond.cleanup3 ]
  %arrayidx22 = getelementptr inbounds i32*, i32** %convolved, i32 %res_y.093
  %tmp3 = load i32*, i32** %arrayidx22, align 4
  br label %for.cond9.preheader.us.us.preheader

for.cond9.preheader.us.us.preheader:              ; preds = %for.cond5.for.cond.cleanup7_crit_edge.us, %for.cond5.preheader.lr.ph
  %res_x.060.us = phi i32 [ %add25.us, %for.cond5.for.cond.cleanup7_crit_edge.us ], [ 0, %for.cond1.preheader ]
  br label %for.cond9.preheader.us.us

for.cond9.preheader.us.us:                        ; preds = %for.cond9.for.cond.cleanup11_crit_edge.us.us, %for.cond9.preheader.us.us.preheader
  %filter_y.056.us.us = phi i32 [ %inc20.us.us, %for.cond9.for.cond.cleanup11_crit_edge.us.us.unr-lcssa ], [ 0, %for.cond9.preheader.us.us.preheader ]
  %result_element.055.us.us = phi i32 [ %add18.us.us.3, %for.cond9.for.cond.cleanup11_crit_edge.us.us.unr-lcssa ], [ 0, %for.cond9.preheader.us.us.preheader ]
  %add.us.us = add i32 %filter_y.056.us.us, %res_y.093
  %arrayidx.us.us = getelementptr inbounds i16*, i16** %filter, i32 %filter_y.056.us.us
  %tmp5 = load i16*, i16** %arrayidx.us.us, align 4
  %arrayidx15.us.us = getelementptr inbounds i16*, i16** %input_image, i32 %add.us.us
  %tmp6 = load i16*, i16** %arrayidx15.us.us, align 4
  br label %for.body12.us.us

for.body12.us.us:                                 ; preds = %for.body12.us.us, %for.cond9.preheader.us.us
  %filter_x.053.us.us = phi i32 [ %inc.us.us.3, %for.body12.us.us ], [ 0, %for.cond9.preheader.us.us ]
  %result_element.152.us.us = phi i32 [ %add18.us.us.3, %for.body12.us.us ], [ %result_element.055.us.us, %for.cond9.preheader.us.us ]
  %niter = phi i32 [ %niter.nsub.3, %for.body12.us.us ], [ %unroll_iter, %for.cond9.preheader.us.us ]
  %add13.us.us = add i32 %filter_x.053.us.us, %res_x.060.us
  %arrayidx14.us.us = getelementptr inbounds i16, i16* %tmp5, i32 %filter_x.053.us.us
  %tmp9 = load i16, i16* %arrayidx14.us.us, align 2
  %conv.us.us = sext i16 %tmp9 to i32
  %arrayidx16.us.us = getelementptr inbounds i16, i16* %tmp6, i32 %add13.us.us
  %tmp10 = load i16, i16* %arrayidx16.us.us, align 2
  %conv17.us.us = sext i16 %tmp10 to i32
  %mul.us.us = mul nsw i32 %conv17.us.us, %conv.us.us
  %add18.us.us = add nsw i32 %mul.us.us, %result_element.152.us.us
  %inc.us.us = or i32 %filter_x.053.us.us, 1
  %add13.us.us.1 = add i32 %inc.us.us, %res_x.060.us
  %arrayidx14.us.us.1 = getelementptr inbounds i16, i16* %tmp5, i32 %inc.us.us
  %tmp11 = load i16, i16* %arrayidx14.us.us.1, align 2
  %conv.us.us.1 = sext i16 %tmp11 to i32
  %arrayidx16.us.us.1 = getelementptr inbounds i16, i16* %tmp6, i32 %add13.us.us.1
  %tmp12 = load i16, i16* %arrayidx16.us.us.1, align 2
  %conv17.us.us.1 = sext i16 %tmp12 to i32
  %mul.us.us.1 = mul nsw i32 %conv17.us.us.1, %conv.us.us.1
  %add18.us.us.1 = add nsw i32 %mul.us.us.1, %add18.us.us
  %inc.us.us.1 = or i32 %filter_x.053.us.us, 2
  %add13.us.us.2 = add i32 %inc.us.us.1, %res_x.060.us
  %arrayidx14.us.us.2 = getelementptr inbounds i16, i16* %tmp5, i32 %inc.us.us.1
  %tmp13 = load i16, i16* %arrayidx14.us.us.2, align 2
  %conv.us.us.2 = sext i16 %tmp13 to i32
  %arrayidx16.us.us.2 = getelementptr inbounds i16, i16* %tmp6, i32 %add13.us.us.2
  %tmp14 = load i16, i16* %arrayidx16.us.us.2, align 2
  %conv17.us.us.2 = sext i16 %tmp14 to i32
  %mul.us.us.2 = mul nsw i32 %conv17.us.us.2, %conv.us.us.2
  %add18.us.us.2 = add nsw i32 %mul.us.us.2, %add18.us.us.1
  %inc.us.us.2 = or i32 %filter_x.053.us.us, 3
  %add13.us.us.3 = add i32 %inc.us.us.2, %res_x.060.us
  %arrayidx14.us.us.3 = getelementptr inbounds i16, i16* %tmp5, i32 %inc.us.us.2
  %tmp15 = load i16, i16* %arrayidx14.us.us.3, align 2
  %conv.us.us.3 = sext i16 %tmp15 to i32
  %arrayidx16.us.us.3 = getelementptr inbounds i16, i16* %tmp6, i32 %add13.us.us.3
  %tmp16 = load i16, i16* %arrayidx16.us.us.3, align 2
  %conv17.us.us.3 = sext i16 %tmp16 to i32
  %mul.us.us.3 = mul nsw i32 %conv17.us.us.3, %conv.us.us.3
  %add18.us.us.3 = add nsw i32 %mul.us.us.3, %add18.us.us.2
  %inc.us.us.3 = add i32 %filter_x.053.us.us, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond9.for.cond.cleanup11_crit_edge.us.us.unr-lcssa, label %for.body12.us.us

for.cond9.for.cond.cleanup11_crit_edge.us.us.unr-lcssa: ; preds = %for.body12.us.us, %for.cond9.preheader.us.us
  %inc20.us.us = add nuw i32 %filter_y.056.us.us, 1
  %exitcond98 = icmp eq i32 %inc20.us.us, %filter_dim
  br i1 %exitcond98, label %for.cond5.for.cond.cleanup7_crit_edge.us, label %for.cond9.preheader.us.us

for.cond5.for.cond.cleanup7_crit_edge.us:         ; preds = %for.cond9.for.cond.cleanup11_crit_edge.us.us
  %arrayidx23.us = getelementptr inbounds i32, i32* %tmp3, i32 %res_x.060.us
  store i32 %add18.us.us.3, i32* %arrayidx23.us, align 4
  %add25.us = add nuw i32 %res_x.060.us, 1
  %exitcond99 = icmp eq i32 %add25.us, %out_width
  br i1 %exitcond99, label %for.cond.cleanup3, label %for.cond9.preheader.us.us.preheader

for.cond.cleanup3:                                ; preds = %for.cond5.for.cond.cleanup7_crit_edge.us, %for.cond5.preheader.preheader, %for.cond1.preheader
  %add28 = add nuw i32 %res_y.093, 1
  %exitcond100 = icmp eq i32 %add28, %out_height
  br i1 %exitcond100, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void
}

; CHECK-LABEL: mul_8x8
; CHECK: @ %for.body

; CHECK-DEFAULT: str{{.*}}, #16]!
; CHECK-DEFAULT: ldrb{{.*}}, #4]!
; CHECK-DEFAULT: ldrb{{.*}}, #4]!

; CHECK-COMPLEX: str{{.*}}, #16]!
; CHECK-COMPLEX: ldrb{{.*}}, #4]!
; CHECK-COMPLEX: ldrb{{.*}}, #4]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body.epil
; CHECK-T2: ldrb{{.*}}, #1]!
; CHECK-T2: ldrb{{.*}}, #1]!
; CHECK-T2: str{{.*}}, #4]!

define void @mul_8x8(i8* nocapture readonly %A, i8* nocapture readonly %B, i32* nocapture %C, i32 %N) {
entry:
  %cmp9 = icmp eq i32 %N, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = add i32 %N, -1
  %xtraiter = and i32 %N, 3
  %tmp1 = icmp ult i32 %tmp, 3
  br i1 %tmp1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = sub i32 %N, %xtraiter
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %i.010.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa
  %i.010.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.010.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %arrayidx.epil = getelementptr inbounds i8, i8* %A, i32 %i.010.epil
  %tmp2 = load i8, i8* %arrayidx.epil, align 1
  %conv.epil = zext i8 %tmp2 to i32
  %arrayidx1.epil = getelementptr inbounds i8, i8* %B, i32 %i.010.epil
  %tmp3 = load i8, i8* %arrayidx1.epil, align 1
  %conv2.epil = zext i8 %tmp3 to i32
  %mul.epil = mul nuw nsw i32 %conv2.epil, %conv.epil
  %arrayidx3.epil = getelementptr inbounds i32, i32* %C, i32 %i.010.epil
  store i32 %mul.epil, i32* %arrayidx3.epil, align 4
  %inc.epil = add nuw i32 %i.010.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.010 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i32 %i.010
  %tmp4 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %tmp4 to i32
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.010
  %tmp5 = load i8, i8* %arrayidx1, align 1
  %conv2 = zext i8 %tmp5 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %i.010
  store i32 %mul, i32* %arrayidx3, align 4
  %inc = or i32 %i.010, 1
  %arrayidx.1 = getelementptr inbounds i8, i8* %A, i32 %inc
  %tmp6 = load i8, i8* %arrayidx.1, align 1
  %conv.1 = zext i8 %tmp6 to i32
  %arrayidx1.1 = getelementptr inbounds i8, i8* %B, i32 %inc
  %tmp7 = load i8, i8* %arrayidx1.1, align 1
  %conv2.1 = zext i8 %tmp7 to i32
  %mul.1 = mul nuw nsw i32 %conv2.1, %conv.1
  %arrayidx3.1 = getelementptr inbounds i32, i32* %C, i32 %inc
  store i32 %mul.1, i32* %arrayidx3.1, align 4
  %inc.1 = or i32 %i.010, 2
  %arrayidx.2 = getelementptr inbounds i8, i8* %A, i32 %inc.1
  %tmp8 = load i8, i8* %arrayidx.2, align 1
  %conv.2 = zext i8 %tmp8 to i32
  %arrayidx1.2 = getelementptr inbounds i8, i8* %B, i32 %inc.1
  %tmp9 = load i8, i8* %arrayidx1.2, align 1
  %conv2.2 = zext i8 %tmp9 to i32
  %mul.2 = mul nuw nsw i32 %conv2.2, %conv.2
  %arrayidx3.2 = getelementptr inbounds i32, i32* %C, i32 %inc.1
  store i32 %mul.2, i32* %arrayidx3.2, align 4
  %inc.2 = or i32 %i.010, 3
  %arrayidx.3 = getelementptr inbounds i8, i8* %A, i32 %inc.2
  %tmp10 = load i8, i8* %arrayidx.3, align 1
  %conv.3 = zext i8 %tmp10 to i32
  %arrayidx1.3 = getelementptr inbounds i8, i8* %B, i32 %inc.2
  %tmp11 = load i8, i8* %arrayidx1.3, align 1
  %conv2.3 = zext i8 %tmp11 to i32
  %mul.3 = mul nuw nsw i32 %conv2.3, %conv.3
  %arrayidx3.3 = getelementptr inbounds i32, i32* %C, i32 %inc.2
  store i32 %mul.3, i32* %arrayidx3.3, align 4
  %inc.3 = add i32 %i.010, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

; CHECK-LABEL: mul_16x8
; CHECK: @ %for.body 

; CHECK-DEFAULT: str{{.*}}, #16]!
; CHECK-DEFAULT: ldrsh{{.*}}, #8]!

; CHECK-COMPLEX: ldrsh{{.*}}, #8]!
; CHECK-COMPLEX: str{{.*}}, #16]!
; CHECK-COMPLEX: ldrb{{.*}}, #4]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body.epil
; CHECK-T2: ldrsh{{.*}}, #2]!
; CHECK-T2: ldrb{{.*}}, #1]!
; CHECK-T2: str{{.*}}, #4]!

define void @mul_16x8(i16* nocapture readonly %A, i8* nocapture readonly %B, i32* nocapture %C, i32 %N) {
entry:
  %cmp9 = icmp eq i32 %N, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = add i32 %N, -1
  %xtraiter = and i32 %N, 3
  %tmp1 = icmp ult i32 %tmp, 3
  br i1 %tmp1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = sub i32 %N, %xtraiter
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %i.010.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa
  %i.010.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.010.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %arrayidx.epil = getelementptr inbounds i16, i16* %A, i32 %i.010.epil
  %tmp2 = load i16, i16* %arrayidx.epil, align 2
  %conv.epil = sext i16 %tmp2 to i32
  %arrayidx1.epil = getelementptr inbounds i8, i8* %B, i32 %i.010.epil
  %tmp3 = load i8, i8* %arrayidx1.epil, align 1
  %conv2.epil = zext i8 %tmp3 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %arrayidx3.epil = getelementptr inbounds i32, i32* %C, i32 %i.010.epil
  store i32 %mul.epil, i32* %arrayidx3.epil, align 4
  %inc.epil = add nuw i32 %i.010.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.010 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %i.010
  %tmp4 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %tmp4 to i32
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.010
  %tmp5 = load i8, i8* %arrayidx1, align 1
  %conv2 = zext i8 %tmp5 to i32
  %mul = mul nsw i32 %conv2, %conv
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %i.010
  store i32 %mul, i32* %arrayidx3, align 4
  %inc = or i32 %i.010, 1
  %arrayidx.1 = getelementptr inbounds i16, i16* %A, i32 %inc
  %tmp6 = load i16, i16* %arrayidx.1, align 2
  %conv.1 = sext i16 %tmp6 to i32
  %arrayidx1.1 = getelementptr inbounds i8, i8* %B, i32 %inc
  %tmp7 = load i8, i8* %arrayidx1.1, align 1
  %conv2.1 = zext i8 %tmp7 to i32
  %mul.1 = mul nsw i32 %conv2.1, %conv.1
  %arrayidx3.1 = getelementptr inbounds i32, i32* %C, i32 %inc
  store i32 %mul.1, i32* %arrayidx3.1, align 4
  %inc.1 = or i32 %i.010, 2
  %arrayidx.2 = getelementptr inbounds i16, i16* %A, i32 %inc.1
  %tmp8 = load i16, i16* %arrayidx.2, align 2
  %conv.2 = sext i16 %tmp8 to i32
  %arrayidx1.2 = getelementptr inbounds i8, i8* %B, i32 %inc.1
  %tmp9 = load i8, i8* %arrayidx1.2, align 1
  %conv2.2 = zext i8 %tmp9 to i32
  %mul.2 = mul nsw i32 %conv2.2, %conv.2
  %arrayidx3.2 = getelementptr inbounds i32, i32* %C, i32 %inc.1
  store i32 %mul.2, i32* %arrayidx3.2, align 4
  %inc.2 = or i32 %i.010, 3
  %arrayidx.3 = getelementptr inbounds i16, i16* %A, i32 %inc.2
  %tmp10 = load i16, i16* %arrayidx.3, align 2
  %conv.3 = sext i16 %tmp10 to i32
  %arrayidx1.3 = getelementptr inbounds i8, i8* %B, i32 %inc.2
  %tmp11 = load i8, i8* %arrayidx1.3, align 1
  %conv2.3 = zext i8 %tmp11 to i32
  %mul.3 = mul nsw i32 %conv2.3, %conv.3
  %arrayidx3.3 = getelementptr inbounds i32, i32* %C, i32 %inc.2
  store i32 %mul.3, i32* %arrayidx3.3, align 4
  %inc.3 = add i32 %i.010, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

; CHECK-LABEL: mul_16x16
; CHECK: @ %for.body

; TODO: pre-indexed loads
; CHECK-DEFAULT-NOT: ldrsh{{.*}}]!
; CHECK-DEFAULT: str{{.*}}, #16]!
; CHECK-DEFAULT-NOT: ldrsh{{.*}}]!

; CHECK-COMPLEX: ldrsh{{.*}}]!
; CHECK-COMPLEX: ldrsh{{.*}}]!
; CHECK-COMPLEX: str{{.*}}]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body.epil
; CHECK-T2: ldrsh{{.*}}, #2]!
; CHECK-T2: ldrsh{{.*}}, #2]!
; CHECK-T2: str{{.*}}, #4]!

define void @mul_16x16(i16* nocapture readonly %A, i16* nocapture readonly %B, i32* nocapture %C, i32 %N) {
entry:
  %cmp9 = icmp eq i32 %N, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = add i32 %N, -1
  %xtraiter = and i32 %N, 3
  %tmp1 = icmp ult i32 %tmp, 3
  br i1 %tmp1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = sub i32 %N, %xtraiter
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %i.010.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa
  %i.010.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.010.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %arrayidx.epil = getelementptr inbounds i16, i16* %A, i32 %i.010.epil
  %tmp2 = load i16, i16* %arrayidx.epil, align 2
  %conv.epil = sext i16 %tmp2 to i32
  %arrayidx1.epil = getelementptr inbounds i16, i16* %B, i32 %i.010.epil
  %tmp3 = load i16, i16* %arrayidx1.epil, align 2
  %conv2.epil = sext i16 %tmp3 to i32
  %mul.epil = mul nsw i32 %conv2.epil, %conv.epil
  %arrayidx3.epil = getelementptr inbounds i32, i32* %C, i32 %i.010.epil
  store i32 %mul.epil, i32* %arrayidx3.epil, align 4
  %inc.epil = add nuw i32 %i.010.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.010 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %i.010
  %tmp4 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %tmp4 to i32
  %arrayidx1 = getelementptr inbounds i16, i16* %B, i32 %i.010
  %tmp5 = load i16, i16* %arrayidx1, align 2
  %conv2 = sext i16 %tmp5 to i32
  %mul = mul nsw i32 %conv2, %conv
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i32 %i.010
  store i32 %mul, i32* %arrayidx3, align 4
  %inc = or i32 %i.010, 1
  %arrayidx.1 = getelementptr inbounds i16, i16* %A, i32 %inc
  %tmp6 = load i16, i16* %arrayidx.1, align 2
  %conv.1 = sext i16 %tmp6 to i32
  %arrayidx1.1 = getelementptr inbounds i16, i16* %B, i32 %inc
  %tmp7 = load i16, i16* %arrayidx1.1, align 2
  %conv2.1 = sext i16 %tmp7 to i32
  %mul.1 = mul nsw i32 %conv2.1, %conv.1
  %arrayidx3.1 = getelementptr inbounds i32, i32* %C, i32 %inc
  store i32 %mul.1, i32* %arrayidx3.1, align 4
  %inc.1 = or i32 %i.010, 2
  %arrayidx.2 = getelementptr inbounds i16, i16* %A, i32 %inc.1
  %tmp8 = load i16, i16* %arrayidx.2, align 2
  %conv.2 = sext i16 %tmp8 to i32
  %arrayidx1.2 = getelementptr inbounds i16, i16* %B, i32 %inc.1
  %tmp9 = load i16, i16* %arrayidx1.2, align 2
  %conv2.2 = sext i16 %tmp9 to i32
  %mul.2 = mul nsw i32 %conv2.2, %conv.2
  %arrayidx3.2 = getelementptr inbounds i32, i32* %C, i32 %inc.1
  store i32 %mul.2, i32* %arrayidx3.2, align 4
  %inc.2 = or i32 %i.010, 3
  %arrayidx.3 = getelementptr inbounds i16, i16* %A, i32 %inc.2
  %tmp10 = load i16, i16* %arrayidx.3, align 2
  %conv.3 = sext i16 %tmp10 to i32
  %arrayidx1.3 = getelementptr inbounds i16, i16* %B, i32 %inc.2
  %tmp11 = load i16, i16* %arrayidx1.3, align 2
  %conv2.3 = sext i16 %tmp11 to i32
  %mul.3 = mul nsw i32 %conv2.3, %conv.3
  %arrayidx3.3 = getelementptr inbounds i32, i32* %C, i32 %inc.2
  store i32 %mul.3, i32* %arrayidx3.3, align 4
  %inc.3 = add i32 %i.010, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

; CHECK-LABEL: mul_8x8_2d
; CHECK: @ %for.body4.us

; CHECK-DEFAULT: ldr{{.*}}, #16]!
; CHECK-DEFAULT: ldrb{{.*}}, #4]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body4.us.epil
; CHECK-T2: ldrb{{.*}}, #1]!
; CHECK-T2: ldr{{.*}}, #4]!

define void @mul_8x8_2d(i8* nocapture readonly %A, i8** nocapture readonly %B, i32** nocapture readonly %C, i32 %N, i32 %M) {
entry:
  %cmp24 = icmp eq i32 %N, 0
  %cmp222 = icmp eq i32 %M, 0
  %or.cond = or i1 %cmp24, %cmp222
  br i1 %or.cond, label %for.cond.cleanup, label %for.cond1.preheader.us.preheader

for.cond1.preheader.us.preheader:                 ; preds = %entry
  %tmp = add i32 %M, -1
  %xtraiter = and i32 %M, 3
  %tmp1 = icmp ult i32 %tmp, 3
  %unroll_iter = sub i32 %M, %xtraiter
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %i.025.us = phi i32 [ %inc11.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %arrayidx.us = getelementptr inbounds i8, i8* %A, i32 %i.025.us
  %arrayidx5.us = getelementptr inbounds i8*, i8** %B, i32 %i.025.us
  %arrayidx8.us = getelementptr inbounds i32*, i32** %C, i32 %i.025.us
  %.pre = load i8*, i8** %arrayidx5.us, align 4
  %.pre30 = load i32*, i32** %arrayidx8.us, align 4
  br i1 %tmp1, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %j.023.us = phi i32 [ %inc.us.3, %for.body4.us ], [ 0, %for.cond1.preheader.us ]
  %niter = phi i32 [ %niter.nsub.3, %for.body4.us ], [ %unroll_iter, %for.cond1.preheader.us ]
  %tmp2 = load i8, i8* %arrayidx.us, align 1
  %conv.us = zext i8 %tmp2 to i32
  %arrayidx6.us = getelementptr inbounds i8, i8* %.pre, i32 %j.023.us
  %tmp3 = load i8, i8* %arrayidx6.us, align 1
  %conv7.us = zext i8 %tmp3 to i32
  %mul.us = mul nuw nsw i32 %conv7.us, %conv.us
  %arrayidx9.us = getelementptr inbounds i32, i32* %.pre30, i32 %j.023.us
  %tmp4 = load i32, i32* %arrayidx9.us, align 4
  %add.us = add nsw i32 %tmp4, %mul.us
  store i32 %add.us, i32* %arrayidx9.us, align 4
  %inc.us = or i32 %j.023.us, 1
  %tmp5 = load i8, i8* %arrayidx.us, align 1
  %conv.us.1 = zext i8 %tmp5 to i32
  %arrayidx6.us.1 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us
  %tmp6 = load i8, i8* %arrayidx6.us.1, align 1
  %conv7.us.1 = zext i8 %tmp6 to i32
  %mul.us.1 = mul nuw nsw i32 %conv7.us.1, %conv.us.1
  %arrayidx9.us.1 = getelementptr inbounds i32, i32* %.pre30, i32 %inc.us
  %tmp7 = load i32, i32* %arrayidx9.us.1, align 4
  %add.us.1 = add nsw i32 %tmp7, %mul.us.1
  store i32 %add.us.1, i32* %arrayidx9.us.1, align 4
  %inc.us.1 = or i32 %j.023.us, 2
  %tmp8 = load i8, i8* %arrayidx.us, align 1
  %conv.us.2 = zext i8 %tmp8 to i32
  %arrayidx6.us.2 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us.1
  %tmp9 = load i8, i8* %arrayidx6.us.2, align 1
  %conv7.us.2 = zext i8 %tmp9 to i32
  %mul.us.2 = mul nuw nsw i32 %conv7.us.2, %conv.us.2
  %arrayidx9.us.2 = getelementptr inbounds i32, i32* %.pre30, i32 %inc.us.1
  %tmp10 = load i32, i32* %arrayidx9.us.2, align 4
  %add.us.2 = add nsw i32 %tmp10, %mul.us.2
  store i32 %add.us.2, i32* %arrayidx9.us.2, align 4
  %inc.us.2 = or i32 %j.023.us, 3
  %tmp11 = load i8, i8* %arrayidx.us, align 1
  %conv.us.3 = zext i8 %tmp11 to i32
  %arrayidx6.us.3 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us.2
  %tmp12 = load i8, i8* %arrayidx6.us.3, align 1
  %conv7.us.3 = zext i8 %tmp12 to i32
  %mul.us.3 = mul nuw nsw i32 %conv7.us.3, %conv.us.3
  %arrayidx9.us.3 = getelementptr inbounds i32, i32* %.pre30, i32 %inc.us.2
  %tmp13 = load i32, i32* %arrayidx9.us.3, align 4
  %add.us.3 = add nsw i32 %tmp13, %mul.us.3
  store i32 %add.us.3, i32* %arrayidx9.us.3, align 4
  %inc.us.3 = add i32 %j.023.us, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa: ; preds = %for.body4.us, %for.cond1.preheader.us
  %j.023.us.unr = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us.3, %for.body4.us ]
  br i1 %lcmp.mod, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.body4.us.epil:                                ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %j.023.us.epil = phi i32 [ %inc.us.epil, %for.body4.us.epil ], [ %j.023.us.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body4.us.epil ], [ %xtraiter, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %tmp14 = load i8, i8* %arrayidx.us, align 1
  %conv.us.epil = zext i8 %tmp14 to i32
  %arrayidx6.us.epil = getelementptr inbounds i8, i8* %.pre, i32 %j.023.us.epil
  %tmp15 = load i8, i8* %arrayidx6.us.epil, align 1
  %conv7.us.epil = zext i8 %tmp15 to i32
  %mul.us.epil = mul nuw nsw i32 %conv7.us.epil, %conv.us.epil
  %arrayidx9.us.epil = getelementptr inbounds i32, i32* %.pre30, i32 %j.023.us.epil
  %tmp16 = load i32, i32* %arrayidx9.us.epil, align 4
  %add.us.epil = add nsw i32 %tmp16, %mul.us.epil
  store i32 %add.us.epil, i32* %arrayidx9.us.epil, align 4
  %inc.us.epil = add nuw i32 %j.023.us.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %inc11.us = add nuw i32 %i.025.us, 1
  %exitcond28 = icmp eq i32 %inc11.us, %N
  br i1 %exitcond28, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  ret void
}

; CHECK-LABEL: mul_16x16_2d
; CHECK: @ %for.body4.us

; CHECK-DEFAULT: ldr{{.*}}, #16]!
; CHECK-DEFAULT: ldrsh{{.*}}, #8]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body4.us.epil
; CHECK-T2: ldrsh{{.*}}, #2]!
; CHECK-T2: ldr{{.*}}, #4]!

define void @mul_16x16_2d(i16* nocapture readonly %A, i16** nocapture readonly %B, i32** nocapture readonly %C, i32 %N, i32 %M) {
entry:
  %cmp24 = icmp eq i32 %N, 0
  %cmp222 = icmp eq i32 %M, 0
  %or.cond = or i1 %cmp24, %cmp222
  br i1 %or.cond, label %for.cond.cleanup, label %for.cond1.preheader.us.preheader

for.cond1.preheader.us.preheader:                 ; preds = %entry
  %tmp = add i32 %M, -1
  %xtraiter = and i32 %M, 3
  %tmp1 = icmp ult i32 %tmp, 3
  %unroll_iter = sub i32 %M, %xtraiter
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %i.025.us = phi i32 [ %inc11.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %i.025.us
  %tmp2 = load i16, i16* %arrayidx.us, align 2
  %conv.us = sext i16 %tmp2 to i32
  %arrayidx5.us = getelementptr inbounds i16*, i16** %B, i32 %i.025.us
  %tmp3 = load i16*, i16** %arrayidx5.us, align 4
  %arrayidx8.us = getelementptr inbounds i32*, i32** %C, i32 %i.025.us
  %tmp4 = load i32*, i32** %arrayidx8.us, align 4
  br i1 %tmp1, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %j.023.us = phi i32 [ %inc.us.3, %for.body4.us ], [ 0, %for.cond1.preheader.us ]
  %niter = phi i32 [ %niter.nsub.3, %for.body4.us ], [ %unroll_iter, %for.cond1.preheader.us ]
  %arrayidx6.us = getelementptr inbounds i16, i16* %tmp3, i32 %j.023.us
  %tmp5 = load i16, i16* %arrayidx6.us, align 2
  %conv7.us = sext i16 %tmp5 to i32
  %mul.us = mul nsw i32 %conv7.us, %conv.us
  %arrayidx9.us = getelementptr inbounds i32, i32* %tmp4, i32 %j.023.us
  %tmp6 = load i32, i32* %arrayidx9.us, align 4
  %add.us = add nsw i32 %tmp6, %mul.us
  store i32 %add.us, i32* %arrayidx9.us, align 4
  %inc.us = or i32 %j.023.us, 1
  %arrayidx6.us.1 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us
  %tmp7 = load i16, i16* %arrayidx6.us.1, align 2
  %conv7.us.1 = sext i16 %tmp7 to i32
  %mul.us.1 = mul nsw i32 %conv7.us.1, %conv.us
  %arrayidx9.us.1 = getelementptr inbounds i32, i32* %tmp4, i32 %inc.us
  %tmp8 = load i32, i32* %arrayidx9.us.1, align 4
  %add.us.1 = add nsw i32 %tmp8, %mul.us.1
  store i32 %add.us.1, i32* %arrayidx9.us.1, align 4
  %inc.us.1 = or i32 %j.023.us, 2
  %arrayidx6.us.2 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us.1
  %tmp9 = load i16, i16* %arrayidx6.us.2, align 2
  %conv7.us.2 = sext i16 %tmp9 to i32
  %mul.us.2 = mul nsw i32 %conv7.us.2, %conv.us
  %arrayidx9.us.2 = getelementptr inbounds i32, i32* %tmp4, i32 %inc.us.1
  %tmp10 = load i32, i32* %arrayidx9.us.2, align 4
  %add.us.2 = add nsw i32 %tmp10, %mul.us.2
  store i32 %add.us.2, i32* %arrayidx9.us.2, align 4
  %inc.us.2 = or i32 %j.023.us, 3
  %arrayidx6.us.3 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us.2
  %tmp11 = load i16, i16* %arrayidx6.us.3, align 2
  %conv7.us.3 = sext i16 %tmp11 to i32
  %mul.us.3 = mul nsw i32 %conv7.us.3, %conv.us
  %arrayidx9.us.3 = getelementptr inbounds i32, i32* %tmp4, i32 %inc.us.2
  %tmp12 = load i32, i32* %arrayidx9.us.3, align 4
  %add.us.3 = add nsw i32 %tmp12, %mul.us.3
  store i32 %add.us.3, i32* %arrayidx9.us.3, align 4
  %inc.us.3 = add i32 %j.023.us, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa: ; preds = %for.body4.us, %for.cond1.preheader.us
  %j.023.us.unr = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us.3, %for.body4.us ]
  br i1 %lcmp.mod, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.body4.us.epil:                                ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %j.023.us.epil = phi i32 [ %inc.us.epil, %for.body4.us.epil ], [ %j.023.us.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body4.us.epil ], [ %xtraiter, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %arrayidx6.us.epil = getelementptr inbounds i16, i16* %tmp3, i32 %j.023.us.epil
  %tmp13 = load i16, i16* %arrayidx6.us.epil, align 2
  %conv7.us.epil = sext i16 %tmp13 to i32
  %mul.us.epil = mul nsw i32 %conv7.us.epil, %conv.us
  %arrayidx9.us.epil = getelementptr inbounds i32, i32* %tmp4, i32 %j.023.us.epil
  %tmp14 = load i32, i32* %arrayidx9.us.epil, align 4
  %add.us.epil = add nsw i32 %tmp14, %mul.us.epil
  store i32 %add.us.epil, i32* %arrayidx9.us.epil, align 4
  %inc.us.epil = add nuw i32 %j.023.us.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %inc11.us = add nuw i32 %i.025.us, 1
  %exitcond28 = icmp eq i32 %inc11.us, %N
  br i1 %exitcond28, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  ret void
}

; CHECK-LABEL: mac_8x8_2d
; CHECK: @ %for.body4.us

; TODO: Both input arrays could use pre-indexed loads.
; TODO: pre-indexed stores.
; CHECK-DEFAULT: ldrb{{.*}}, #4]!
; CHECK-DEFAULT-NOT: ldr{{.*}}]!
; CHECK-DEFAULT-NOT: str{{.*}}]!

; TODO: Increased complexity shouldn't prevent indexed accesses.
; CHECK-COMPLEX-NOT: ldr{{.*}}]!
; CHECK-COMPLEX-NOT: str{{.*}}]!

; DISABLED-NOT: ldr{{.*}}]!
; DISABLED-NOT: str{{.*}}]!

; CHECK-T2: @ %for.body4.us.epil
; CHECK-T2: ldrb{{.*}}, #1]!

define void @mac_8x8_2d(i8* nocapture readonly %A, i8** nocapture readonly %B, i32* nocapture %C, i32 %N, i32 %M) {
entry:
  %cmp22 = icmp eq i32 %N, 0
  %cmp220 = icmp eq i32 %M, 0
  %or.cond = or i1 %cmp22, %cmp220
  br i1 %or.cond, label %for.cond.cleanup, label %for.cond1.preheader.us.preheader

for.cond1.preheader.us.preheader:                 ; preds = %entry
  %tmp = add i32 %M, -1
  %xtraiter = and i32 %M, 3
  %tmp1 = icmp ult i32 %tmp, 3
  %unroll_iter = sub i32 %M, %xtraiter
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %i.023.us = phi i32 [ %inc10.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %arrayidx.us = getelementptr inbounds i8, i8* %A, i32 %i.023.us
  %arrayidx5.us = getelementptr inbounds i8*, i8** %B, i32 %i.023.us
  %arrayidx8.us = getelementptr inbounds i32, i32* %C, i32 %i.023.us
  %.pre = load i8*, i8** %arrayidx5.us, align 4
  %.pre28 = load i32, i32* %arrayidx8.us, align 4
  br i1 %tmp1, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %tmp2 = phi i32 [ %add.us.3, %for.body4.us ], [ %.pre28, %for.cond1.preheader.us ]
  %j.021.us = phi i32 [ %inc.us.3, %for.body4.us ], [ 0, %for.cond1.preheader.us ]
  %niter = phi i32 [ %niter.nsub.3, %for.body4.us ], [ %unroll_iter, %for.cond1.preheader.us ]
  %tmp3 = load i8, i8* %arrayidx.us, align 1
  %conv.us = zext i8 %tmp3 to i32
  %arrayidx6.us = getelementptr inbounds i8, i8* %.pre, i32 %j.021.us
  %tmp4 = load i8, i8* %arrayidx6.us, align 1
  %conv7.us = zext i8 %tmp4 to i32
  %mul.us = mul nuw nsw i32 %conv7.us, %conv.us
  %add.us = add nsw i32 %mul.us, %tmp2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %inc.us = or i32 %j.021.us, 1
  %tmp5 = load i8, i8* %arrayidx.us, align 1
  %conv.us.1 = zext i8 %tmp5 to i32
  %arrayidx6.us.1 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us
  %tmp6 = load i8, i8* %arrayidx6.us.1, align 1
  %conv7.us.1 = zext i8 %tmp6 to i32
  %mul.us.1 = mul nuw nsw i32 %conv7.us.1, %conv.us.1
  %add.us.1 = add nsw i32 %mul.us.1, %add.us
  store i32 %add.us.1, i32* %arrayidx8.us, align 4
  %inc.us.1 = or i32 %j.021.us, 2
  %tmp7 = load i8, i8* %arrayidx.us, align 1
  %conv.us.2 = zext i8 %tmp7 to i32
  %arrayidx6.us.2 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us.1
  %tmp8 = load i8, i8* %arrayidx6.us.2, align 1
  %conv7.us.2 = zext i8 %tmp8 to i32
  %mul.us.2 = mul nuw nsw i32 %conv7.us.2, %conv.us.2
  %add.us.2 = add nsw i32 %mul.us.2, %add.us.1
  store i32 %add.us.2, i32* %arrayidx8.us, align 4
  %inc.us.2 = or i32 %j.021.us, 3
  %tmp9 = load i8, i8* %arrayidx.us, align 1
  %conv.us.3 = zext i8 %tmp9 to i32
  %arrayidx6.us.3 = getelementptr inbounds i8, i8* %.pre, i32 %inc.us.2
  %tmp10 = load i8, i8* %arrayidx6.us.3, align 1
  %conv7.us.3 = zext i8 %tmp10 to i32
  %mul.us.3 = mul nuw nsw i32 %conv7.us.3, %conv.us.3
  %add.us.3 = add nsw i32 %mul.us.3, %add.us.2
  store i32 %add.us.3, i32* %arrayidx8.us, align 4
  %inc.us.3 = add i32 %j.021.us, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa: ; preds = %for.body4.us, %for.cond1.preheader.us
  %.unr = phi i32 [ %.pre28, %for.cond1.preheader.us ], [ %add.us.3, %for.body4.us ]
  %j.021.us.unr = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us.3, %for.body4.us ]
  br i1 %lcmp.mod, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.body4.us.epil:                                ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %tmp11 = phi i32 [ %add.us.epil, %for.body4.us.epil ], [ %.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %j.021.us.epil = phi i32 [ %inc.us.epil, %for.body4.us.epil ], [ %j.021.us.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body4.us.epil ], [ %xtraiter, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %tmp12 = load i8, i8* %arrayidx.us, align 1
  %conv.us.epil = zext i8 %tmp12 to i32
  %arrayidx6.us.epil = getelementptr inbounds i8, i8* %.pre, i32 %j.021.us.epil
  %tmp13 = load i8, i8* %arrayidx6.us.epil, align 1
  %conv7.us.epil = zext i8 %tmp13 to i32
  %mul.us.epil = mul nuw nsw i32 %conv7.us.epil, %conv.us.epil
  %add.us.epil = add nsw i32 %mul.us.epil, %tmp11
  store i32 %add.us.epil, i32* %arrayidx8.us, align 4
  %inc.us.epil = add nuw i32 %j.021.us.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %inc10.us = add nuw i32 %i.023.us, 1
  %exitcond26 = icmp eq i32 %inc10.us, %N
  br i1 %exitcond26, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  ret void
}

; CHECK-LABEL: mac_16x16_2d
; CHECK: @ %for.body4.us

; TODO: pre-indexed loads for both input arrays.
; CHECK-DEFAULT: ldrsh{{.*}}, #8]!
; CHECK-DEFAULT-NOT: ldr{{.*}}]!

; TODO: increased complexity should lead to better codegen.
; CHECK-COMPLEX-NOT: ldr{{.*}}]!

; DISABLED-NOT: ldr{{.*}}]!

; CHECK-T2: @ %for.body4.us.epil
; CHECK-T2: ldrsh{{.*}}, #2]!

define void @mac_16x16_2d(i16* nocapture readonly %A, i16** nocapture readonly %B, i32* nocapture %C, i32 %N, i32 %M) {
entry:
  %cmp23 = icmp eq i32 %N, 0
  %cmp220 = icmp eq i32 %M, 0
  %or.cond = or i1 %cmp23, %cmp220
  br i1 %or.cond, label %for.cond.cleanup, label %for.cond1.preheader.us.preheader

for.cond1.preheader.us.preheader:                 ; preds = %entry
  %tmp = add i32 %M, -1
  %xtraiter = and i32 %M, 3
  %tmp1 = icmp ult i32 %tmp, 3
  %unroll_iter = sub i32 %M, %xtraiter
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %for.cond1.preheader.us.preheader
  %i.024.us = phi i32 [ %inc10.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %i.024.us
  %tmp2 = load i16, i16* %arrayidx.us, align 2
  %conv.us = sext i16 %tmp2 to i32
  %arrayidx5.us = getelementptr inbounds i16*, i16** %B, i32 %i.024.us
  %tmp3 = load i16*, i16** %arrayidx5.us, align 4
  %arrayidx8.us = getelementptr inbounds i32, i32* %C, i32 %i.024.us
  %arrayidx8.promoted.us = load i32, i32* %arrayidx8.us, align 4
  br i1 %tmp1, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.cond1.preheader.us
  %add22.us = phi i32 [ %add.us.3, %for.body4.us ], [ %arrayidx8.promoted.us, %for.cond1.preheader.us ]
  %j.021.us = phi i32 [ %inc.us.3, %for.body4.us ], [ 0, %for.cond1.preheader.us ]
  %niter = phi i32 [ %niter.nsub.3, %for.body4.us ], [ %unroll_iter, %for.cond1.preheader.us ]
  %arrayidx6.us = getelementptr inbounds i16, i16* %tmp3, i32 %j.021.us
  %tmp4 = load i16, i16* %arrayidx6.us, align 2
  %conv7.us = sext i16 %tmp4 to i32
  %mul.us = mul nsw i32 %conv7.us, %conv.us
  %add.us = add nsw i32 %mul.us, %add22.us
  %inc.us = or i32 %j.021.us, 1
  %arrayidx6.us.1 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us
  %tmp5 = load i16, i16* %arrayidx6.us.1, align 2
  %conv7.us.1 = sext i16 %tmp5 to i32
  %mul.us.1 = mul nsw i32 %conv7.us.1, %conv.us
  %add.us.1 = add nsw i32 %mul.us.1, %add.us
  %inc.us.1 = or i32 %j.021.us, 2
  %arrayidx6.us.2 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us.1
  %tmp6 = load i16, i16* %arrayidx6.us.2, align 2
  %conv7.us.2 = sext i16 %tmp6 to i32
  %mul.us.2 = mul nsw i32 %conv7.us.2, %conv.us
  %add.us.2 = add nsw i32 %mul.us.2, %add.us.1
  %inc.us.2 = or i32 %j.021.us, 3
  %arrayidx6.us.3 = getelementptr inbounds i16, i16* %tmp3, i32 %inc.us.2
  %tmp7 = load i16, i16* %arrayidx6.us.3, align 2
  %conv7.us.3 = sext i16 %tmp7 to i32
  %mul.us.3 = mul nsw i32 %conv7.us.3, %conv.us
  %add.us.3 = add nsw i32 %mul.us.3, %add.us.2
  %inc.us.3 = add i32 %j.021.us, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa, label %for.body4.us

for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa: ; preds = %for.body4.us, %for.cond1.preheader.us
  %add.us.lcssa.ph = phi i32 [ undef, %for.cond1.preheader.us ], [ %add.us.3, %for.body4.us ]
  %add22.us.unr = phi i32 [ %arrayidx8.promoted.us, %for.cond1.preheader.us ], [ %add.us.3, %for.body4.us ]
  %j.021.us.unr = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us.3, %for.body4.us ]
  br i1 %lcmp.mod, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.body4.us.epil:                                ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %add22.us.epil = phi i32 [ %add.us.epil, %for.body4.us.epil ], [ %add22.us.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %j.021.us.epil = phi i32 [ %inc.us.epil, %for.body4.us.epil ], [ %j.021.us.unr, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body4.us.epil ], [ %xtraiter, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ]
  %arrayidx6.us.epil = getelementptr inbounds i16, i16* %tmp3, i32 %j.021.us.epil
  %tmp8 = load i16, i16* %arrayidx6.us.epil, align 2
  %conv7.us.epil = sext i16 %tmp8 to i32
  %mul.us.epil = mul nsw i32 %conv7.us.epil, %conv.us
  %add.us.epil = add nsw i32 %mul.us.epil, %add22.us.epil
  %inc.us.epil = add nuw i32 %j.021.us.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond1.for.cond.cleanup3_crit_edge.us, label %for.body4.us.epil

for.cond1.for.cond.cleanup3_crit_edge.us:         ; preds = %for.body4.us.epil, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa
  %add.us.lcssa = phi i32 [ %add.us.lcssa.ph, %for.cond1.for.cond.cleanup3_crit_edge.us.unr-lcssa ], [ %add.us.epil, %for.body4.us.epil ]
  store i32 %add.us.lcssa, i32* %arrayidx8.us, align 4
  %inc10.us = add nuw i32 %i.024.us, 1
  %exitcond27 = icmp eq i32 %inc10.us, %N
  br i1 %exitcond27, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  ret void
}

; CHECK-LABEL: mul32x32_backwards
; CHECK: @ %for.body

; TODO: post increments for decreasing addresses
; CHECK-DEFAULT-NOT: ldr{{.*}}]!
; CHECK-DEFAULT-NOT: str{{.*}}]!

; CHECK-COMPLEX-NOT: ldr{{.*}}]!
; CHECK-COMPLEX-NOT: str{{.*}}]!

define void @mul32x32_backwards(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %i.08 = add i32 %N, -1
  %cmp9 = icmp sgt i32 %i.08, -1
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %xtraiter = and i32 %N, 3
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.body.prol.loopexit, label %for.body.prol

for.body.prol:                                    ; preds = %for.body.prol, %for.body.preheader
  %i.010.prol = phi i32 [ %i.0.prol, %for.body.prol ], [ %i.08, %for.body.preheader ]
  %prol.iter = phi i32 [ %prol.iter.sub, %for.body.prol ], [ %xtraiter, %for.body.preheader ]
  %arrayidx.prol = getelementptr inbounds i32, i32* %b, i32 %i.010.prol
  %tmp = load i32, i32* %arrayidx.prol, align 4
  %arrayidx1.prol = getelementptr inbounds i32, i32* %c, i32 %i.010.prol
  %tmp1 = load i32, i32* %arrayidx1.prol, align 4
  %mul.prol = mul nsw i32 %tmp1, %tmp
  %arrayidx2.prol = getelementptr inbounds i32, i32* %a, i32 %i.010.prol
  store i32 %mul.prol, i32* %arrayidx2.prol, align 4
  %i.0.prol = add i32 %i.010.prol, -1
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.prol.loopexit, label %for.body.prol

for.body.prol.loopexit:                           ; preds = %for.body.prol, %for.body.preheader
  %i.010.unr = phi i32 [ %i.08, %for.body.preheader ], [ %i.0.prol, %for.body.prol ]
  %tmp2 = icmp ult i32 %i.08, 3
  br i1 %tmp2, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %for.body.prol.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.prol.loopexit
  %i.010 = phi i32 [ %i.0.3, %for.body ], [ %i.010.unr, %for.body.prol.loopexit ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.010
  %tmp3 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i32 %i.010
  %tmp4 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %tmp4, %tmp3
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %i.010
  store i32 %mul, i32* %arrayidx2, align 4
  %i.0 = add i32 %i.010, -1
  %arrayidx.1 = getelementptr inbounds i32, i32* %b, i32 %i.0
  %tmp5 = load i32, i32* %arrayidx.1, align 4
  %arrayidx1.1 = getelementptr inbounds i32, i32* %c, i32 %i.0
  %tmp6 = load i32, i32* %arrayidx1.1, align 4
  %mul.1 = mul nsw i32 %tmp6, %tmp5
  %arrayidx2.1 = getelementptr inbounds i32, i32* %a, i32 %i.0
  store i32 %mul.1, i32* %arrayidx2.1, align 4
  %i.0.1 = add i32 %i.010, -2
  %arrayidx.2 = getelementptr inbounds i32, i32* %b, i32 %i.0.1
  %tmp7 = load i32, i32* %arrayidx.2, align 4
  %arrayidx1.2 = getelementptr inbounds i32, i32* %c, i32 %i.0.1
  %tmp8 = load i32, i32* %arrayidx1.2, align 4
  %mul.2 = mul nsw i32 %tmp8, %tmp7
  %arrayidx2.2 = getelementptr inbounds i32, i32* %a, i32 %i.0.1
  store i32 %mul.2, i32* %arrayidx2.2, align 4
  %i.0.2 = add i32 %i.010, -3
  %arrayidx.3 = getelementptr inbounds i32, i32* %b, i32 %i.0.2
  %tmp9 = load i32, i32* %arrayidx.3, align 4
  %arrayidx1.3 = getelementptr inbounds i32, i32* %c, i32 %i.0.2
  %tmp10 = load i32, i32* %arrayidx1.3, align 4
  %mul.3 = mul nsw i32 %tmp10, %tmp9
  %arrayidx2.3 = getelementptr inbounds i32, i32* %a, i32 %i.0.2
  store i32 %mul.3, i32* %arrayidx2.3, align 4
  %i.0.3 = add i32 %i.010, -4
  %cmp.3 = icmp sgt i32 %i.0.3, -1
  br i1 %cmp.3, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: mul32x32_forwards
; CHECK: @ %for.body

; TODO: Would be good for the complexity limit didn't have to be increased to
; enable the pre-indexed accesses.

; CHECK-DEFAULT-NOT: ldr{{.*}}]!
; CHECK-DEFAULT-NOT: str{{.*}}]!

; CHECK-COMPLEX: ldr{{.*}}, #16]!
; CHECK-COMPLEX: ldr{{.*}}, #16]!
; CHECK-COMPLEX: str{{.*}}, #16]!

; CHECK-T2: @ %for.body.epil
; CHECK-T2: ldr{{.*}}, #4]!
; CHECK-T2: ldr{{.*}}, #4]!
; CHECK-T2: str{{.*}}, #4]!

define void @mul32x32_forwards(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = add i32 %N, -1
  %xtraiter = and i32 %N, 3
  %tmp1 = icmp ult i32 %tmp, 3
  br i1 %tmp1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = sub i32 %N, %xtraiter
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %i.09.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa
  %i.09.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %i.09.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body.epil ], [ %xtraiter, %for.cond.cleanup.loopexit.unr-lcssa ]
  %arrayidx.epil = getelementptr inbounds i32, i32* %b, i32 %i.09.epil
  %tmp2 = load i32, i32* %arrayidx.epil, align 4
  %arrayidx1.epil = getelementptr inbounds i32, i32* %c, i32 %i.09.epil
  %tmp3 = load i32, i32* %arrayidx1.epil, align 4
  %mul.epil = mul nsw i32 %tmp3, %tmp2
  %arrayidx2.epil = getelementptr inbounds i32, i32* %a, i32 %i.09.epil
  store i32 %mul.epil, i32* %arrayidx2.epil, align 4
  %inc.epil = add nuw nsw i32 %i.09.epil, 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.cond.cleanup, label %for.body.epil

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.09 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.09
  %tmp4 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i32 %i.09
  %tmp5 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %tmp5, %tmp4
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %i.09
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = or i32 %i.09, 1
  %arrayidx.1 = getelementptr inbounds i32, i32* %b, i32 %inc
  %tmp6 = load i32, i32* %arrayidx.1, align 4
  %arrayidx1.1 = getelementptr inbounds i32, i32* %c, i32 %inc
  %tmp7 = load i32, i32* %arrayidx1.1, align 4
  %mul.1 = mul nsw i32 %tmp7, %tmp6
  %arrayidx2.1 = getelementptr inbounds i32, i32* %a, i32 %inc
  store i32 %mul.1, i32* %arrayidx2.1, align 4
  %inc.1 = or i32 %i.09, 2
  %arrayidx.2 = getelementptr inbounds i32, i32* %b, i32 %inc.1
  %tmp8 = load i32, i32* %arrayidx.2, align 4
  %arrayidx1.2 = getelementptr inbounds i32, i32* %c, i32 %inc.1
  %tmp9 = load i32, i32* %arrayidx1.2, align 4
  %mul.2 = mul nsw i32 %tmp9, %tmp8
  %arrayidx2.2 = getelementptr inbounds i32, i32* %a, i32 %inc.1
  store i32 %mul.2, i32* %arrayidx2.2, align 4
  %inc.2 = or i32 %i.09, 3
  %arrayidx.3 = getelementptr inbounds i32, i32* %b, i32 %inc.2
  %tmp10 = load i32, i32* %arrayidx.3, align 4
  %arrayidx1.3 = getelementptr inbounds i32, i32* %c, i32 %inc.2
  %tmp11 = load i32, i32* %arrayidx1.3, align 4
  %mul.3 = mul nsw i32 %tmp11, %tmp10
  %arrayidx2.3 = getelementptr inbounds i32, i32* %a, i32 %inc.2
  store i32 %mul.3, i32* %arrayidx2.3, align 4
  %inc.3 = add nuw nsw i32 %i.09, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}
