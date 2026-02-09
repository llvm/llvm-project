; RUN: llc -mtriple=aarch64-none-linux-gnu -start-before=aarch64-isel %s -o /dev/null
; Regression test for AArch64 compile-time regression, referring to PR #166962.
source_filename = "third_party/tensorflow/core/kernels/image/resize_bicubic_op.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-grtev4-linux-gnu"

%"struct.std::__u::array" = type { [4 x i64] }
%"class.tensorflow::(anonymous namespace)::CachedInterpolationCalculator" = type { [4 x i64] }
%"struct.tensorflow::InitOnStartupMarker" = type { i8 }

declare void @llvm.lifetime.start.p0(ptr captures(none))

declare void @llvm.lifetime.end.p0(ptr captures(none))

define fastcc void @_ZN10tensorflow12_GLOBAL__N_125ComputeXWeightsAndIndicesERKNS_17ImageResizerStateEbPNSt3__u6vectorINS0_17WeightsAndIndicesENS4_9allocatorIS6_EEEE(ptr noundef nonnull readonly captures(none) %x_wais) {
entry:
  %new_x_indices.i116 = alloca %"struct.std::__u::array", align 8
  %new_x_indices.i = alloca %"struct.std::__u::array", align 8
  %calc = alloca %"class.tensorflow::(anonymous namespace)::CachedInterpolationCalculator", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %calc)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %calc, i8 -1, i64 32, i1 false)
  %0 = load i64, ptr null, align 8
  br i1 false, label %for.cond.preheader, label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %entry
  %1 = icmp sgt i64 %0, 0
  br label %for.body8

for.cond.preheader:                               ; preds = %entry
  %2 = icmp sgt i64 %0, 0
  %sunkaddr = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val102 = load i64, ptr %sunkaddr, align 8
  %cmp.i = icmp ult i64 0, %x_wais.val102
  %x_wais.val101 = load ptr, ptr %x_wais, align 8
  %3 = load i64, ptr null, align 8
  %4 = load float, ptr null, align 4
  %scevgep230 = getelementptr i8, ptr %x_wais.val101, i64 0
  tail call fastcc void null(float noundef %4, i64 noundef 0, i64 noundef %3, ptr noundef nonnull %scevgep230)
  %sunkaddr256 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val100 = load i64, ptr %sunkaddr256, align 8
  %cmp.i103 = icmp ult i64 0, %x_wais.val100
  %x_wais.val99 = load ptr, ptr %x_wais, align 8
  %scevgep228 = getelementptr i8, ptr %x_wais.val99, i64 0
  %scevgep229 = getelementptr i8, ptr %scevgep228, i64 16
  %5 = load i64, ptr %scevgep229, align 8
  %scevgep226 = getelementptr i8, ptr %x_wais.val99, i64 0
  %scevgep227 = getelementptr i8, ptr %scevgep226, i64 24
  %6 = load i64, ptr %scevgep227, align 8
  %scevgep224 = getelementptr i8, ptr %x_wais.val99, i64 0
  %scevgep225 = getelementptr i8, ptr %scevgep224, i64 32
  %7 = load i64, ptr %scevgep225, align 8
  %scevgep222 = getelementptr i8, ptr %x_wais.val99, i64 0
  %scevgep223 = getelementptr i8, ptr %scevgep222, i64 40
  %8 = load i64, ptr %scevgep223, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %new_x_indices.i)
  store i64 %5, ptr %new_x_indices.i, align 8
  %sunkaddr257 = getelementptr inbounds i8, ptr %new_x_indices.i, i64 8
  store i64 %6, ptr %sunkaddr257, align 8
  %sunkaddr258 = getelementptr inbounds i8, ptr %new_x_indices.i, i64 16
  store i64 %7, ptr %sunkaddr258, align 8
  %sunkaddr259 = getelementptr inbounds i8, ptr %new_x_indices.i, i64 24
  store i64 %8, ptr %sunkaddr259, align 8
  %9 = load i64, ptr %calc, align 8
  %cmp4.not.i = icmp eq i64 %9, %5
  %sunkaddr260 = getelementptr inbounds i8, ptr %calc, i64 8
  %10 = load i64, ptr %sunkaddr260, align 8
  %cmp4.152.i = icmp eq i64 %10, %5
  store i64 %5, ptr %calc, align 8
  br label %if.end.1.i

if.end.1.i:                                       ; preds = %for.cond.preheader
  %new_indices_hand.15361.i = phi i64 [ 1, %for.cond.preheader ]
  %sunkaddr261 = getelementptr inbounds i8, ptr %calc, i64 16
  %11 = load i64, ptr %sunkaddr261, align 8
  %arrayidx.i.2.i = getelementptr inbounds nuw i64, ptr %new_x_indices.i, i64 %new_indices_hand.15361.i
  %12 = load i64, ptr %arrayidx.i.2.i, align 8
  %cmp4.2.i = icmp eq i64 %11, %12
  %cmp5.2.i = icmp samesign ult i64 %new_indices_hand.15361.i, 2
  %arrayidx12.2.i = getelementptr inbounds nuw i64, ptr %calc, i64 %new_indices_hand.15361.i
  store i64 %11, ptr %arrayidx12.2.i, align 8
  %inc13.2.i = add nuw nsw i64 %new_indices_hand.15361.i, 1
  %arrayidx.i.3.i.phi.trans.insert = getelementptr inbounds nuw i64, ptr %new_x_indices.i, i64 %inc13.2.i
  %.pre189 = load i64, ptr %arrayidx.i.3.i.phi.trans.insert, align 8
  %sunkaddr262 = getelementptr inbounds i8, ptr %calc, i64 24
  %13 = load i64, ptr %sunkaddr262, align 8
  %cmp4.3.i = icmp eq i64 %13, %.pre189
  %cmp5.3.i = icmp samesign ult i64 %inc13.2.i, 3
  %arrayidx12.3.i = getelementptr inbounds nuw i64, ptr %calc, i64 %inc13.2.i
  store i64 %.pre189, ptr %arrayidx12.3.i, align 8
  %inc13.3.i = add nuw nsw i64 %inc13.2.i, 1
  %cond = icmp eq i64 %inc13.3.i, 2
  call void @llvm.lifetime.end.p0(ptr nonnull %new_x_indices.i)
  %scevgep220 = getelementptr i8, ptr %x_wais.val99, i64 0
  %scevgep221 = getelementptr i8, ptr %scevgep220, i64 48
  %14 = trunc i64 %inc13.3.i to i32
  store i32 %14, ptr %scevgep221, align 8
  %inc = add nuw nsw i64 0, 1
  %15 = load i64, ptr null, align 8
  %lsr.iv.next219 = add nuw i64 0, 56
  %cmp = icmp slt i64 %inc, %15
  %cmp26184 = icmp sgt i64 %15, 0
  br label %for.body28

for.body8:                                        ; preds = %for.body8, %for.cond4.preheader
  %lsr.iv231 = phi i64 [ 48, %for.cond4.preheader ], [ %lsr.iv.next232, %for.body8 ]
  %x3.0181 = phi i64 [ 0, %for.cond4.preheader ], [ %inc21, %for.body8 ]
  %16 = load i64, ptr null, align 8
  %sunkaddr268 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val98 = load i64, ptr %sunkaddr268, align 8
  %cmp.i107 = icmp ult i64 %x3.0181, %x_wais.val98
  %x_wais.val97 = load ptr, ptr %x_wais, align 8
  %17 = load float, ptr null, align 4
  %scevgep252 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %tmp = trunc i64 %x3.0181 to i32
  %conv.i.i = sitofp i32 %tmp to float
  %mul.i.i = fmul float %17, %conv.i.i
  %18 = tail call noundef float @llvm.floor.f32(float %mul.i.i)
  %conv2.i = fptosi float %18 to i64
  %conv3.i = sitofp i64 %conv2.i to float
  %sub.i = fsub float %mul.i.i, %conv3.i
  %mul.i = fmul float %sub.i, 1.024000e+03
  %call4.i = tail call i64 null(float noundef %mul.i)
  %19 = load atomic i8, ptr getelementptr inbounds (<{ %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", [2 x i8], ptr, i64, ptr, i64 }>, ptr null, i32 0, i32 18) acquire, align 8
  %20 = zext i8 %19 to i32
  %21 = and i32 %20, 1
  %guard.uninitialized1.i.i = icmp eq i32 %21, 0
  %22 = tail call i32 null(ptr nonnull getelementptr inbounds (<{ %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", [2 x i8], ptr, i64, ptr, i64 }>, ptr null, i32 0, i32 18))
  %tobool3.not.i.i = icmp eq i32 %22, 0
  %call5.i.i = tail call fastcc noundef ptr null(double noundef -7.500000e-01)
  store ptr %call5.i.i, ptr getelementptr inbounds (<{ %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", [2 x i8], ptr, i64, ptr, i64 }>, ptr null, i32 0, i32 17), align 8
  tail call void null(ptr nonnull getelementptr inbounds (<{ %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", [2 x i8], ptr, i64, ptr, i64 }>, ptr null, i32 0, i32 18))
  %retval.0.i.i = load ptr, ptr getelementptr inbounds (<{ %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", %"struct.tensorflow::InitOnStartupMarker", [2 x i8], ptr, i64, ptr, i64 }>, ptr null, i32 0, i32 17), align 8
  %mul6.i = shl nsw i64 %call4.i, 1
  %23 = getelementptr float, ptr %retval.0.i.i, i64 %mul6.i
  %arrayidx.i111 = getelementptr i8, ptr %23, i64 4
  %24 = load float, ptr %arrayidx.i111, align 4
  %sunkaddr270 = getelementptr i8, ptr %scevgep252, i64 -48
  store float %24, ptr %sunkaddr270, align 8
  %25 = load float, ptr %23, align 4
  %scevgep250 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %scevgep251 = getelementptr i8, ptr %scevgep250, i64 -44
  store float %25, ptr %scevgep251, align 4
  %mul10.i = sub i64 2048, %mul6.i
  %arrayidx11.i = getelementptr inbounds float, ptr %retval.0.i.i, i64 %mul10.i
  %26 = load float, ptr %arrayidx11.i, align 4
  %scevgep248 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %scevgep249 = getelementptr i8, ptr %scevgep248, i64 -40
  store float %26, ptr %scevgep249, align 8
  %add14.i = sub i64 2049, %mul6.i
  %arrayidx15.i = getelementptr inbounds float, ptr %retval.0.i.i, i64 %add14.i
  %27 = load float, ptr %arrayidx15.i, align 4
  %scevgep246 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %scevgep247 = getelementptr i8, ptr %scevgep246, i64 -36
  store float %27, ptr %scevgep247, align 4
  %sub.i.i = add nsw i64 %16, -1
  %28 = insertelement <2 x i64> poison, i64 %conv2.i, i64 0
  %29 = shufflevector <2 x i64> %28, <2 x i64> poison, <2 x i32> zeroinitializer
  %30 = tail call <2 x i64> @llvm.smax.v2i64(<2 x i64> %29, <2 x i64> <i64 1, i64 0>)
  %scevgep244 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %scevgep245 = getelementptr i8, ptr %scevgep244, i64 -32
  %31 = add nsw <2 x i64> %30, <i64 -1, i64 0>
  %32 = insertelement <2 x i64> poison, i64 %sub.i.i, i64 0
  %33 = shufflevector <2 x i64> %32, <2 x i64> poison, <2 x i32> zeroinitializer
  %34 = tail call <2 x i64> @llvm.smin.v2i64(<2 x i64> %31, <2 x i64> %33)
  store <2 x i64> %34, ptr %scevgep245, align 8
  %35 = tail call <2 x i64> @llvm.smax.v2i64(<2 x i64> %29, <2 x i64> <i64 -1, i64 -2>)
  %scevgep242 = getelementptr i8, ptr %x_wais.val97, i64 %lsr.iv231
  %scevgep243 = getelementptr i8, ptr %scevgep242, i64 -16
  %36 = add nsw <2 x i64> %35, <i64 1, i64 2>
  %37 = tail call <2 x i64> @llvm.smin.v2i64(<2 x i64> %36, <2 x i64> %33)
  store <2 x i64> %37, ptr %scevgep243, align 8
  %sunkaddr271 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val96 = load i64, ptr %sunkaddr271, align 8
  %cmp.i112 = icmp ult i64 %x3.0181, %x_wais.val96
  %x_wais.val95 = load ptr, ptr %x_wais, align 8
  %scevgep240 = getelementptr i8, ptr %x_wais.val95, i64 %lsr.iv231
  %scevgep241 = getelementptr i8, ptr %scevgep240, i64 -32
  %38 = load i64, ptr %scevgep241, align 8
  %scevgep238 = getelementptr i8, ptr %x_wais.val95, i64 %lsr.iv231
  %scevgep239 = getelementptr i8, ptr %scevgep238, i64 -24
  %39 = load i64, ptr %scevgep239, align 8
  %scevgep236 = getelementptr i8, ptr %x_wais.val95, i64 %lsr.iv231
  %scevgep237 = getelementptr i8, ptr %scevgep236, i64 -16
  %40 = load i64, ptr %scevgep237, align 8
  %scevgep234 = getelementptr i8, ptr %x_wais.val95, i64 %lsr.iv231
  %scevgep235 = getelementptr i8, ptr %scevgep234, i64 -8
  %41 = load i64, ptr %scevgep235, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %new_x_indices.i116)
  store i64 %38, ptr %new_x_indices.i116, align 8
  %sunkaddr272 = getelementptr inbounds i8, ptr %new_x_indices.i116, i64 8
  store i64 %39, ptr %sunkaddr272, align 8
  %sunkaddr273 = getelementptr inbounds i8, ptr %new_x_indices.i116, i64 16
  store i64 %40, ptr %sunkaddr273, align 8
  %sunkaddr274 = getelementptr inbounds i8, ptr %new_x_indices.i116, i64 24
  store i64 %41, ptr %sunkaddr274, align 8
  %42 = load i64, ptr %calc, align 8
  %cmp4.not.i120 = icmp eq i64 %42, %38
  %sunkaddr275 = getelementptr inbounds i8, ptr %calc, i64 8
  %43 = load i64, ptr %sunkaddr275, align 8
  %cmp4.1.i161 = icmp eq i64 %43, %39
  %sunkaddr276 = getelementptr inbounds i8, ptr %calc, i64 16
  %44 = load i64, ptr %sunkaddr276, align 8
  %arrayidx.i.2.i128 = getelementptr inbounds nuw i64, ptr %new_x_indices.i116, i64 2
  %45 = load i64, ptr %arrayidx.i.2.i128, align 8
  %cmp4.2.i129 = icmp eq i64 %44, %45
  %cmp5.2.i149 = icmp samesign ult i64 2, 2
  %arrayidx12.2.i154 = getelementptr inbounds nuw i64, ptr %calc, i64 2
  store i64 %44, ptr %arrayidx12.2.i154, align 8
  %inc13.2.i151 = add nuw nsw i64 2, 1
  %arrayidx.i.3.i134.phi.trans.insert = getelementptr inbounds nuw i64, ptr %new_x_indices.i116, i64 %inc13.2.i151
  %.pre = load i64, ptr %arrayidx.i.3.i134.phi.trans.insert, align 8
  %sunkaddr277 = getelementptr inbounds i8, ptr %calc, i64 24
  %46 = load i64, ptr %sunkaddr277, align 8
  %cmp4.3.i135 = icmp eq i64 %46, %.pre
  %cmp5.3.i143 = icmp samesign ult i64 %inc13.2.i151, 3
  %arrayidx12.3.i147 = getelementptr inbounds nuw i64, ptr %calc, i64 %inc13.2.i151
  store i64 %.pre, ptr %arrayidx12.3.i147, align 8
  %inc13.3.i145 = add nuw nsw i64 %inc13.2.i151, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %new_x_indices.i116)
  %scevgep233 = getelementptr i8, ptr %x_wais.val95, i64 %lsr.iv231
  %47 = trunc i64 %inc13.3.i145 to i32
  store i32 %47, ptr %scevgep233, align 8
  %inc21 = add nuw nsw i64 %x3.0181, 1
  %48 = load i64, ptr null, align 8
  %lsr.iv.next232 = add i64 %lsr.iv231, 56
  %cmp6 = icmp slt i64 %inc21, %48
  br label %for.body8

for.body28:                                       ; preds = %for.body28, %if.end.1.i
  %lsr.iv = phi i64 [ 0, %if.end.1.i ], [ %lsr.iv.next, %for.body28 ]
  %indvars.iv = phi i64 [ 0, %if.end.1.i ], [ %indvars.iv.next, %for.body28 ]
  %sunkaddr282 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val94 = load i64, ptr %sunkaddr282, align 8
  %cmp.i163 = icmp ugt i64 %x_wais.val94, %indvars.iv
  %x_wais.val93 = load ptr, ptr %x_wais, align 8
  %49 = load i64, ptr null, align 8
  %scevgep216 = getelementptr i8, ptr %x_wais.val93, i64 %lsr.iv
  %scevgep217 = getelementptr i8, ptr %scevgep216, i64 16
  %50 = load i64, ptr %scevgep217, align 8
  %mul = mul nsw i64 %50, %49
  store i64 %mul, ptr %scevgep217, align 8
  %sunkaddr284 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val92 = load i64, ptr %sunkaddr284, align 8
  %cmp.i167 = icmp ugt i64 %x_wais.val92, %indvars.iv
  %x_wais.val91 = load ptr, ptr %x_wais, align 8
  %51 = load i64, ptr null, align 8
  %scevgep214 = getelementptr i8, ptr %x_wais.val91, i64 %lsr.iv
  %scevgep215 = getelementptr i8, ptr %scevgep214, i64 24
  %52 = load i64, ptr %scevgep215, align 8
  %mul36 = mul nsw i64 %52, %51
  store i64 %mul36, ptr %scevgep215, align 8
  %sunkaddr286 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val90 = load i64, ptr %sunkaddr286, align 8
  %cmp.i171 = icmp ugt i64 %x_wais.val90, %indvars.iv
  %x_wais.val89 = load ptr, ptr %x_wais, align 8
  %53 = load i64, ptr null, align 8
  %scevgep212 = getelementptr i8, ptr %x_wais.val89, i64 %lsr.iv
  %scevgep213 = getelementptr i8, ptr %scevgep212, i64 32
  %54 = load i64, ptr %scevgep213, align 8
  %mul41 = mul nsw i64 %54, %53
  store i64 %mul41, ptr %scevgep213, align 8
  %sunkaddr288 = getelementptr i8, ptr %x_wais, i64 8
  %x_wais.val88 = load i64, ptr %sunkaddr288, align 8
  %cmp.i175 = icmp ugt i64 %x_wais.val88, %indvars.iv
  %x_wais.val = load ptr, ptr %x_wais, align 8
  %55 = load i64, ptr null, align 8
  %scevgep = getelementptr i8, ptr %x_wais.val, i64 %lsr.iv
  %scevgep211 = getelementptr i8, ptr %scevgep, i64 40
  %56 = load i64, ptr %scevgep211, align 8
  %mul46 = mul nsw i64 %56, %55
  store i64 %mul46, ptr %scevgep211, align 8
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %57 = load i64, ptr null, align 8
  %lsr.iv.next = add nuw i64 %lsr.iv, 56
  %cmp26 = icmp sgt i64 %57, %indvars.iv.next
  br label %for.body28
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare float @llvm.floor.f32(float)

declare <2 x i64> @llvm.smax.v2i64(<2 x i64>, <2 x i64>)

declare <2 x i64> @llvm.smin.v2i64(<2 x i64>, <2 x i64>)
