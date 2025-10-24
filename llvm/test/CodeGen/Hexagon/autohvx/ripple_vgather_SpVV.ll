; Verify that we generate HVX vgather for the given input.
; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b,-long-calls -hexagon-allow-scatter-gather-hvx < %s | FileCheck %s
; CHECK-LABEL: SpVV_Ripple:
; CHECK: vtmp.h = vgather(r{{[0-9]+}},m0,v{{[0-9]+}}.h).h
; CHECK: vmem(r0+#0) = vtmp.new

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local i32 @SpVV_Ripple(ptr nocapture noundef writeonly %scratchpad, ptr nocapture noundef readonly %Source_1, ptr nocapture noundef readonly %S_index, i32 noundef %nS, ptr nocapture noundef readonly %Source_2) local_unnamed_addr #1 {
entry:
  %Source_2.ripple.bcast.splatinsert = insertelement <64 x ptr> poison, ptr %Source_2, i64 0
  %Source_2.ripple.bcast.splat = shufflevector <64 x ptr> %Source_2.ripple.bcast.splatinsert, <64 x ptr> poison, <64 x i32> zeroinitializer
  %div16 = lshr i32 %nS, 6
  %cmp6.not = icmp ult i32 %nS, 64
  br i1 %cmp6.not, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %lsr.iv17 = phi ptr [ %scevgep18, %for.body ], [ %S_index, %entry ]
  %lsr.iv = phi ptr [ %scevgep, %for.body ], [ %Source_1, %entry ]
  %result.08.ripple.vectorized = phi <64 x i32> [ %add8.ripple.vectorized, %for.body ], [ zeroinitializer, %entry ]
  %_ripple_block_0.07 = phi i32 [ %add9, %for.body ], [ 0, %entry ]
  %.ripple.LS.instance = load <64 x i16>, ptr %lsr.iv17, align 2
  %idxprom.ripple.LS.instance = sext <64 x i16> %.ripple.LS.instance to <64 x i32>
  %arrayidx2.ripple.LS.instance = getelementptr inbounds i16, <64 x ptr> %Source_2.ripple.bcast.splat, <64 x i32> %idxprom.ripple.LS.instance
  %.ripple.LS.instance13 = tail call <64 x i16> @llvm.masked.gather.v64i16.v64p0(<64 x ptr> %arrayidx2.ripple.LS.instance, i32 2, <64 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <64 x i16> poison)
  store <64 x i16> %.ripple.LS.instance13, ptr %scratchpad, align 2
  %.ripple.LS.instance15 = load <64 x i16>, ptr %lsr.iv, align 2
  %conv.ripple.LS.instance = sext <64 x i16> %.ripple.LS.instance15 to <64 x i32>
  %conv6.ripple.LS.instance = sext <64 x i16> %.ripple.LS.instance13 to <64 x i32>
  %mul7.ripple.LS.instance = mul nsw <64 x i32> %conv.ripple.LS.instance, %conv6.ripple.LS.instance
  %add8.ripple.vectorized = add <64 x i32> %mul7.ripple.LS.instance, %result.08.ripple.vectorized
  %add9 = add nuw nsw i32 %_ripple_block_0.07, 1
  %scevgep = getelementptr i8, ptr %lsr.iv, i32 128
  %scevgep18 = getelementptr i8, ptr %lsr.iv17, i32 128
  %cmp = icmp ult i32 %add9, %div16
  br i1 %cmp, label %for.body, label %for.end
for.end:                                          ; preds = %for.body, %entry
  %result.0.lcssa.ripple.LS.instance = phi <64 x i32> [ zeroinitializer, %entry ], [ %add8.ripple.vectorized, %for.body ]
  %rdx.shuf = shufflevector <64 x i32> %result.0.lcssa.ripple.LS.instance, <64 x i32> poison, <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx = add <64 x i32> %result.0.lcssa.ripple.LS.instance, %rdx.shuf
  %rdx.shuf19 = shufflevector <64 x i32> %bin.rdx, <64 x i32> poison, <64 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx20 = add <64 x i32> %bin.rdx, %rdx.shuf19
  %rdx.shuf21 = shufflevector <64 x i32> %bin.rdx20, <64 x i32> poison, <64 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx22 = add <64 x i32> %bin.rdx20, %rdx.shuf21
  %rdx.shuf23 = shufflevector <64 x i32> %bin.rdx22, <64 x i32> poison, <64 x i32> <i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx24 = add <64 x i32> %bin.rdx22, %rdx.shuf23
  %rdx.shuf25 = shufflevector <64 x i32> %bin.rdx24, <64 x i32> poison, <64 x i32> <i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx26 = add <64 x i32> %bin.rdx24, %rdx.shuf25
  %rdx.shuf27 = shufflevector <64 x i32> %bin.rdx26, <64 x i32> poison, <64 x i32> <i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx28 = add <64 x i32> %bin.rdx26, %rdx.shuf27
  %0 = extractelement <64 x i32> %bin.rdx28, i32 0
  ret i32 %0
}
