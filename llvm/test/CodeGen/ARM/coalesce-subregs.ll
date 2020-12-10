; RUN: llc < %s -mcpu=cortex-a9 -verify-coalescing -verify-machineinstrs | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios0.0.0"

; CHECK: f
; The vld2 and vst2 are not aligned wrt each other, the second Q loaded is the
; first one stored.
; The coalescer must find a super-register larger than QQ to eliminate the copy
; setting up the vst2 data.
; CHECK: vld2
; CHECK-NOT: vorr
; CHECK-NOT: vmov
; CHECK: vst2
define void @f(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8* %0, i32 4)
  %vld221 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %add.ptr = getelementptr inbounds float, float* %p, i32 8
  %1 = bitcast float* %add.ptr to i8*
  tail call void @llvm.arm.neon.vst2.p0i8.v4f32(i8* %1, <4 x float> %vld221, <4 x float> undef, i32 4)
  ret void
}

; CHECK: f1
; FIXME: This function still has copies.
define void @f1(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8* %0, i32 4)
  %vld221 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %add.ptr = getelementptr inbounds float, float* %p, i32 8
  %1 = bitcast float* %add.ptr to i8*
  %vld22 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8* %1, i32 4)
  %vld2215 = extractvalue { <4 x float>, <4 x float> } %vld22, 0
  tail call void @llvm.arm.neon.vst2.p0i8.v4f32(i8* %1, <4 x float> %vld221, <4 x float> %vld2215, i32 4)
  ret void
}

; CHECK: f2
; FIXME: This function still has copies.
define void @f2(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8* %0, i32 4)
  %vld224 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %qq0.0.1.0 = phi <4 x float> [ %vld224, %entry ], [ %vld2216, %do.body ]
  %c.addr.0 = phi i32 [ %c, %entry ], [ %dec, %do.body ]
  %p.addr.0 = phi float* [ %p, %entry ], [ %add.ptr, %do.body ]
  %add.ptr = getelementptr inbounds float, float* %p.addr.0, i32 8
  %1 = bitcast float* %add.ptr to i8*
  %vld22 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8* %1, i32 4)
  %vld2215 = extractvalue { <4 x float>, <4 x float> } %vld22, 0
  %vld2216 = extractvalue { <4 x float>, <4 x float> } %vld22, 1
  tail call void @llvm.arm.neon.vst2.p0i8.v4f32(i8* %1, <4 x float> %qq0.0.1.0, <4 x float> %vld2215, i32 4)
  %dec = add nsw i32 %c.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret void
}

declare { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32.p0i8(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst2.p0i8.v4f32(i8*, <4 x float>, <4 x float>, i32) nounwind

; CHECK: f3
; This function has lane insertions that span basic blocks.
; The trivial REG_SEQUENCE lowering can't handle that, but the coalescer can.
;
; void f3(float *p, float *q) {
;   float32x2_t x;
;   x[1] = p[3];
;   if (q)
;     x[0] = q[0] + q[1];
;   else
;     x[0] = p[2];
;   vst1_f32(p+4, x);
; }
;
; CHECK-NOT: vmov
; CHECK-NOT: vorr
define void @f3(float* %p, float* %q) nounwind ssp {
entry:
  %arrayidx = getelementptr inbounds float, float* %p, i32 3
  %0 = load float, float* %arrayidx, align 4
  %vecins = insertelement <2 x float> undef, float %0, i32 1
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %1 = load float, float* %q, align 4
  %arrayidx2 = getelementptr inbounds float, float* %q, i32 1
  %2 = load float, float* %arrayidx2, align 4
  %add = fadd float %1, %2
  %vecins3 = insertelement <2 x float> %vecins, float %add, i32 0
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx4 = getelementptr inbounds float, float* %p, i32 2
  %3 = load float, float* %arrayidx4, align 4
  %vecins5 = insertelement <2 x float> %vecins, float %3, i32 0
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %x.0 = phi <2 x float> [ %vecins3, %if.then ], [ %vecins5, %if.else ]
  %add.ptr = getelementptr inbounds float, float* %p, i32 4
  %4 = bitcast float* %add.ptr to i8*
  tail call void @llvm.arm.neon.vst1.p0i8.v2f32(i8* %4, <2 x float> %x.0, i32 4)
  ret void
}

declare void @llvm.arm.neon.vst1.p0i8.v2f32(i8*, <2 x float>, i32) nounwind
declare <2 x float> @llvm.arm.neon.vld1.v2f32.p0i8(i8*, i32) nounwind readonly

; CHECK: f4
; This function inserts a lane into a fully defined vector.
; The destination lane isn't read, so the subregs can coalesce.
; CHECK-NOT: vmov
; CHECK-NOT: vorr
define void @f4(float* %p, float* %q) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld1 = tail call <2 x float> @llvm.arm.neon.vld1.v2f32.p0i8(i8* %0, i32 4)
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %1 = load float, float* %q, align 4
  %arrayidx1 = getelementptr inbounds float, float* %q, i32 1
  %2 = load float, float* %arrayidx1, align 4
  %add = fadd float %1, %2
  %vecins = insertelement <2 x float> %vld1, float %add, i32 1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %x.0 = phi <2 x float> [ %vecins, %if.then ], [ %vld1, %entry ]
  tail call void @llvm.arm.neon.vst1.p0i8.v2f32(i8* %0, <2 x float> %x.0, i32 4)
  ret void
}

; CHECK: f5
; Coalesce vector lanes through phis.
; CHECK: vmov.f32 {{.*}}, #1.0
; CHECK-NOT: vmov
; CHECK-NOT: vorr
; CHECK: bx
; We may leave the last insertelement in the if.end block.
; It is inserting the %add value into a dead lane, but %add causes interference
; in the entry block, and we don't do dead lane checks across basic blocks.
define void @f5(float* %p, float* %q) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* %0, i32 4)
  %vecext = extractelement <4 x float> %vld1, i32 0
  %vecext1 = extractelement <4 x float> %vld1, i32 1
  %vecext2 = extractelement <4 x float> %vld1, i32 2
  %vecext3 = extractelement <4 x float> %vld1, i32 3
  %add = fadd float %vecext3, 1.000000e+00
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds float, float* %q, i32 1
  %1 = load float, float* %arrayidx, align 4
  %add4 = fadd float %vecext, %1
  %2 = load float, float* %q, align 4
  %add6 = fadd float %vecext1, %2
  %arrayidx7 = getelementptr inbounds float, float* %q, i32 2
  %3 = load float, float* %arrayidx7, align 4
  %add8 = fadd float %vecext2, %3
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi float [ %add4, %if.then ], [ %vecext, %entry ]
  %b.0 = phi float [ %add6, %if.then ], [ %vecext1, %entry ]
  %c.0 = phi float [ %add8, %if.then ], [ %vecext2, %entry ]
  %vecinit = insertelement <4 x float> undef, float %a.0, i32 0
  %vecinit9 = insertelement <4 x float> %vecinit, float %b.0, i32 1
  %vecinit10 = insertelement <4 x float> %vecinit9, float %c.0, i32 2
  %vecinit11 = insertelement <4 x float> %vecinit10, float %add, i32 3
  tail call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %0, <4 x float> %vecinit11, i32 4)
  ret void
}

declare <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.p0i8.v4f32(i8*, <4 x float>, i32) nounwind

; CHECK: pr13999
define void @pr13999() nounwind readonly {
entry:
 br i1 true, label %outer_loop, label %loop.end

outer_loop:
 %d = phi double [ 0.0, %entry ], [ %add, %after_inner_loop ]
 %0 = insertelement <2 x double> <double 0.0, double 0.0>, double %d, i32 0
 br i1 undef, label %after_inner_loop, label %inner_loop

inner_loop:
 br i1 true, label %after_inner_loop, label %inner_loop

after_inner_loop:
 %1 = phi <2 x double> [ %0, %outer_loop ], [ <double 0.0, double 0.0>,
%inner_loop ]
 %2 = extractelement <2 x double> %1, i32 1
 %add = fadd double 1.0, %2
 br i1 false, label %loop.end, label %outer_loop

loop.end:
 %d.end = phi double [ 0.0, %entry ], [ %add, %after_inner_loop ]
 ret void
}

; CHECK: pr14078
define arm_aapcs_vfpcc i32 @pr14078(i8* nocapture %arg, i8* nocapture %arg1, i32 %arg2) nounwind uwtable readonly {
bb:
  br i1 undef, label %bb31, label %bb3

bb3:                                              ; preds = %bb12, %bb
  %tmp = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp4 = bitcast <1 x i64> %tmp to <2 x float>
  %tmp5 = shufflevector <2 x float> %tmp4, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp6 = bitcast <4 x float> %tmp5 to <2 x i64>
  %tmp7 = shufflevector <2 x i64> %tmp6, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp8 = bitcast <1 x i64> %tmp7 to <2 x float>
  %tmp9 = tail call <2 x float> @baz(<2 x float> <float 0xFFFFFFFFE0000000, float 0.000000e+00>, <2 x float> %tmp8, <2 x float> zeroinitializer) nounwind
  br i1 undef, label %bb10, label %bb12

bb10:                                             ; preds = %bb3
  %tmp11 = load <4 x float>, <4 x float>* undef, align 8
  br label %bb12

bb12:                                             ; preds = %bb10, %bb3
  %tmp13 = shufflevector <2 x float> %tmp9, <2 x float> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %tmp14 = bitcast <2 x float> %tmp13 to <1 x i64>
  %tmp15 = shufflevector <1 x i64> %tmp14, <1 x i64> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %tmp16 = bitcast <2 x i64> %tmp15 to <4 x float>
  %tmp17 = fmul <4 x float> zeroinitializer, %tmp16
  %tmp18 = bitcast <4 x float> %tmp17 to <2 x i64>
  %tmp19 = shufflevector <2 x i64> %tmp18, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp20 = bitcast <1 x i64> %tmp19 to <2 x float>
  %tmp21 = tail call <2 x float> @baz67(<2 x float> %tmp20, <2 x float> undef) nounwind
  %tmp22 = tail call <2 x float> @baz67(<2 x float> %tmp21, <2 x float> %tmp21) nounwind
  %tmp23 = shufflevector <2 x float> %tmp22, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp24 = bitcast <4 x float> %tmp23 to <2 x i64>
  %tmp25 = shufflevector <2 x i64> %tmp24, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp26 = bitcast <1 x i64> %tmp25 to <2 x float>
  %tmp27 = extractelement <2 x float> %tmp26, i32 0
  %tmp28 = fcmp olt float %tmp27, 0.000000e+00
  %tmp29 = select i1 %tmp28, i32 0, i32 undef
  %tmp30 = icmp ult i32 undef, %arg2
  br i1 %tmp30, label %bb3, label %bb31

bb31:                                             ; preds = %bb12, %bb
  %tmp32 = phi i32 [ 1, %bb ], [ %tmp29, %bb12 ]
  ret i32 %tmp32
}

declare <2 x float> @baz(<2 x float>, <2 x float>, <2 x float>) nounwind readnone

declare <2 x float> @baz67(<2 x float>, <2 x float>) nounwind readnone

%struct.wombat.5 = type { %struct.quux, %struct.quux, %struct.quux, %struct.quux }
%struct.quux = type { <4 x float> }

; CHECK: pr14079
define linkonce_odr arm_aapcs_vfpcc %struct.wombat.5 @pr14079(i8* nocapture %arg, i8* nocapture %arg1, i8* nocapture %arg2) nounwind uwtable inlinehint {
bb:
  %tmp = shufflevector <2 x i64> zeroinitializer, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp3 = bitcast <1 x i64> %tmp to <2 x float>
  %tmp4 = shufflevector <2 x float> %tmp3, <2 x float> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %tmp5 = shufflevector <2 x float> %tmp4, <2 x float> undef, <2 x i32> <i32 1, i32 3>
  %tmp6 = bitcast <2 x float> %tmp5 to <1 x i64>
  %tmp7 = shufflevector <1 x i64> undef, <1 x i64> %tmp6, <2 x i32> <i32 0, i32 1>
  %tmp8 = bitcast <2 x i64> %tmp7 to <4 x float>
  %tmp9 = shufflevector <2 x i64> zeroinitializer, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp10 = bitcast <1 x i64> %tmp9 to <2 x float>
  %tmp11 = shufflevector <2 x float> %tmp10, <2 x float> undef, <2 x i32> <i32 0, i32 2>
  %tmp12 = shufflevector <2 x float> %tmp11, <2 x float> undef, <2 x i32> <i32 0, i32 2>
  %tmp13 = bitcast <2 x float> %tmp12 to <1 x i64>
  %tmp14 = shufflevector <1 x i64> %tmp13, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %tmp15 = bitcast <2 x i64> %tmp14 to <4 x float>
  %tmp16 = insertvalue %struct.wombat.5 undef, <4 x float> %tmp8, 1, 0
  %tmp17 = insertvalue %struct.wombat.5 %tmp16, <4 x float> %tmp15, 2, 0
  %tmp18 = insertvalue %struct.wombat.5 %tmp17, <4 x float> undef, 3, 0
  ret %struct.wombat.5 %tmp18
}

; CHECK: adjustCopiesBackFrom
; The shuffle in if.else3 must be preserved even though adjustCopiesBackFrom
; is tempted to remove it.
; CHECK: vorr d
define internal void @adjustCopiesBackFrom(<2 x i64>* noalias nocapture sret(<2 x i64>) %agg.result, <2 x i64> %in) {
entry:
  %0 = extractelement <2 x i64> %in, i32 0
  %cmp = icmp slt i64 %0, 1
  %.in = select i1 %cmp, <2 x i64> <i64 0, i64 undef>, <2 x i64> %in
  %1 = extractelement <2 x i64> %in, i32 1
  %cmp1 = icmp slt i64 %1, 1
  br i1 %cmp1, label %if.then2, label %if.else3

if.then2:                                         ; preds = %entry
  %2 = insertelement <2 x i64> %.in, i64 0, i32 1
  br label %if.end4

if.else3:                                         ; preds = %entry
  %3 = shufflevector <2 x i64> %.in, <2 x i64> %in, <2 x i32> <i32 0, i32 3>
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %result.2 = phi <2 x i64> [ %2, %if.then2 ], [ %3, %if.else3 ]
  store <2 x i64> %result.2, <2 x i64>* %agg.result, align 128
  ret void
}

; <rdar://problem/12758887>
; RegisterCoalescer::updateRegDefsUses() could visit an instruction more than
; once under rare circumstances. When widening a register from QPR to DTriple
; with the original virtual register in dsub_1_dsub_2, the double rewrite would
; produce an invalid sub-register.
;
; This is because dsub_1_dsub_2 is not an idempotent sub-register index.
; It will translate %vr:dsub_0 -> %vr:dsub_1.
define hidden fastcc void @radar12758887() nounwind optsize ssp {
entry:
  br i1 undef, label %for.body, label %for.end70

for.body:                                         ; preds = %for.end, %entry
  br i1 undef, label %for.body29, label %for.end

for.body29:                                       ; preds = %for.body29, %for.body
  %0 = load <2 x double>, <2 x double>* null, align 1
  %splat40 = shufflevector <2 x double> %0, <2 x double> undef, <2 x i32> zeroinitializer
  %mul41 = fmul <2 x double> undef, %splat40
  %add42 = fadd <2 x double> undef, %mul41
  %splat44 = shufflevector <2 x double> %0, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  %mul45 = fmul <2 x double> undef, %splat44
  %add46 = fadd <2 x double> undef, %mul45
  br i1 undef, label %for.end, label %for.body29

for.end:                                          ; preds = %for.body29, %for.body
  %accumR2.0.lcssa = phi <2 x double> [ zeroinitializer, %for.body ], [ %add42, %for.body29 ]
  %accumI2.0.lcssa = phi <2 x double> [ zeroinitializer, %for.body ], [ %add46, %for.body29 ]
  %1 = shufflevector <2 x double> %accumI2.0.lcssa, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  %add58 = fadd <2 x double> undef, %1
  %mul61 = fmul <2 x double> %add58, undef
  %add63 = fadd <2 x double> undef, %mul61
  %add64 = fadd <2 x double> undef, %add63
  %add67 = fadd <2 x double> undef, %add64
  store <2 x double> %add67, <2 x double>* undef, align 1
  br i1 undef, label %for.end70, label %for.body

for.end70:                                        ; preds = %for.end, %entry
  ret void
}
