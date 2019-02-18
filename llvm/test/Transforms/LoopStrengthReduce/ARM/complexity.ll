target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; RUN: opt -mtriple=thumbv7em %s -S -loop-reduce -lsr-complexity-limit=65536 -o - | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: opt -mtriple=thumbv7em %s -S -loop-reduce -lsr-complexity-limit=2147483647 -o - | FileCheck %s --check-prefix=CHECK-COMPLEX

; CHECK-DEFAULT-LABEL: for.body12.us.us:
; CHECK-DEFAULT: phi i32
; CHECK-DEFAULT: [[LSR_IV:%[^ ]+]] = phi i32 [ [[LSR_IV_NEXT:%[^ ]+]], %for.body12.us.us ], [ 0, %for.cond9.preheader.us.us ]
; CHECK-DEFAULT: phi i32
; CHECK-DEFAULT: [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 8

; CHECK-COMPLEX-LABEL: for.body12.us.us:
; CHECK-COMPLEX: phi i32
; CHECK-COMPLEX: [[LSR_IV6:%[^ ]+]] = phi i16* [ [[SCEVGEP7:%[^ ]+]], %for.body12.us.us ], [ [[SCEVGEP5:%[^ ]+]], %for.cond9.preheader.us.us ]
; CHECK-COMPLEX: [[LSR_IV:%[^ ]+]] = phi i16* [ [[SCEVGEP1:%[^ ]+]], %for.body12.us.us ], [ [[SCEVGEP:%[^ ]+]], %for.cond9.preheader.us.us ]
; CHECK-COMPLEX: phi i32
; CHECK-COMPLEX: [[SCEVGEP1]] = getelementptr i16, i16* [[LSR_IV]], i32 4
; CHECK-COMPLEX: [[SCEVGEP7]] = getelementptr i16, i16* [[LSR_IV6]], i32 4

define void @convolve(i16** nocapture readonly %input_image, i16** nocapture readonly %filter, i32 %filter_dim, i32 %out_width, i32 %out_height, i32** nocapture readonly %convolved) {
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

