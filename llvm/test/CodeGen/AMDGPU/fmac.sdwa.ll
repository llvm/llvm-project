; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1010 %s

; GCN-LABEL: {{^}}addMul2D:
; GFX1010: v_fmac_f16
; GFX1010: v_fmac_f16
define hidden <4 x half> @addMul2D(<4 x i8>* nocapture readonly, float addrspace(4)* nocapture readonly, <2 x i32>, i32) local_unnamed_addr #0 {
  %5 = extractelement <2 x i32> %2, i64 1
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %4
  %8 = extractelement <2 x i32> %2, i64 0
  %9 = icmp sgt i32 %8, 0
  br label %10

10:                                               ; preds = %34, %7
  %11 = phi <4 x half> [ zeroinitializer, %7 ], [ %35, %34 ]
  %12 = phi i32 [ 0, %7 ], [ %36, %34 ]
  br i1 %9, label %13, label %34

13:                                               ; preds = %10
  %14 = mul nsw i32 %12, %3
  %15 = mul nsw i32 %12, %8
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi <4 x half> [ %11, %13 ], [ %31, %16 ]
  %18 = phi i32 [ 0, %13 ], [ %32, %16 ]
  %19 = add nsw i32 %18, %14
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %20
  %22 = load <4 x i8>, <4 x i8>* %21, align 4
  %23 = tail call <4 x half> @_Z13convert_half4Dv4_h(<4 x i8> %22) #8
  %24 = add nsw i32 %18, %15
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds float, float addrspace(4)* %1, i64 %25
  %27 = load float, float addrspace(4)* %26, align 4
  %28 = fptrunc float %27 to half
  %29 = insertelement <4 x half> undef, half %28, i32 0
  %30 = shufflevector <4 x half> %29, <4 x half> undef, <4 x i32> zeroinitializer
  %31 = tail call <4 x half> @llvm.fmuladd.v4f16(<4 x half> %23, <4 x half> %30, <4 x half> %17)
  %32 = add nuw nsw i32 %18, 1
  %33 = icmp eq i32 %32, %8
  br i1 %33, label %34, label %16

34:                                               ; preds = %16, %10
  %35 = phi <4 x half> [ %11, %10 ], [ %31, %16 ]
  %36 = add nuw nsw i32 %12, 1
  %37 = icmp eq i32 %36, %5
  br i1 %37, label %38, label %10

38:                                               ; preds = %34, %4
  %39 = phi <4 x half> [ zeroinitializer, %4 ], [ %35, %34 ]
  ret <4 x half> %39
}

define linkonce_odr hidden <4 x half> @_Z13convert_half4Dv4_h(<4 x i8>) local_unnamed_addr #1 {
  %2 = extractelement <4 x i8> %0, i64 0
  %3 = uitofp i8 %2 to half
  %4 = insertelement <4 x half> undef, half %3, i32 0
  %5 = extractelement <4 x i8> %0, i64 1
  %6 = uitofp i8 %5 to half
  %7 = insertelement <4 x half> %4, half %6, i32 1
  %8 = extractelement <4 x i8> %0, i64 2
  %9 = uitofp i8 %8 to half
  %10 = insertelement <4 x half> %7, half %9, i32 2
  %11 = extractelement <4 x i8> %0, i64 3
  %12 = uitofp i8 %11 to half
  %13 = insertelement <4 x half> %10, half %12, i32 3
  ret <4 x half> %13
}

declare <4 x half> @llvm.fmuladd.v4f16(<4 x half>, <4 x half>, <4 x half>)

attributes #0 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="gfx1010" "target-features"="+16-bit-insts,+dl-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx10-insts,+gfx9-insts,+s-memrealtime,-code-object-v3,-sram-ecc,-xnack" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="64" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+fp64-fp16-denormals,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }
