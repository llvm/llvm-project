; RUN: llc < %s -mcpu=cortex-a8
; Test that this doesn't crash.
; <rdar://problem/12183003>

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.1.0"

declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8.p0(ptr, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.p0.v16i8(ptr, <16 x i8>, i32) nounwind

define void @findEdges(ptr) nounwind ssp {
  %2 = icmp sgt i32 undef, 0
  br i1 %2, label %5, label %3

; <label>:3                                       ; preds = %5, %1
  %4 = phi ptr [ %0, %1 ], [ %19, %5 ]
  ret void

; <label>:5                                       ; preds = %5, %1
  %6 = phi ptr [ %19, %5 ], [ %0, %1 ]
  %7 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8.p0(ptr null, i32 1)
  %8 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %7, 0
  %9 = getelementptr inbounds i8, ptr null, i32 3
  %10 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8.p0(ptr %9, i32 1)
  %11 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %10, 2
  %12 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8.p0(ptr %6, i32 1)
  %13 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %12, 0
  %14 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %12, 1
  %15 = getelementptr inbounds i8, ptr %6, i32 3
  %16 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8.p0(ptr %15, i32 1)
  %17 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %16, 1
  %18 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %16, 2
  %19 = getelementptr inbounds i8, ptr %6, i32 48
  %20 = bitcast <16 x i8> %13 to <2 x i64>
  %21 = bitcast <16 x i8> %8 to <2 x i64>
  %22 = bitcast <16 x i8> %14 to <2 x i64>
  %23 = shufflevector <2 x i64> %22, <2 x i64> undef, <1 x i32> zeroinitializer
  %24 = bitcast <1 x i64> %23 to <8 x i8>
  %25 = zext <8 x i8> %24 to <8 x i16>
  %26 = sub <8 x i16> zeroinitializer, %25
  %27 = bitcast <16 x i8> %17 to <2 x i64>
  %28 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %26) nounwind
  %29 = mul <8 x i16> %28, %28
  %30 = add <8 x i16> zeroinitializer, %29
  %31 = tail call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> undef, <8 x i16> %30) nounwind
  %32 = bitcast <16 x i8> %11 to <2 x i64>
  %33 = shufflevector <2 x i64> %32, <2 x i64> undef, <1 x i32> zeroinitializer
  %34 = bitcast <1 x i64> %33 to <8 x i8>
  %35 = zext <8 x i8> %34 to <8 x i16>
  %36 = sub <8 x i16> %35, zeroinitializer
  %37 = bitcast <16 x i8> %18 to <2 x i64>
  %38 = shufflevector <2 x i64> %37, <2 x i64> undef, <1 x i32> zeroinitializer
  %39 = bitcast <1 x i64> %38 to <8 x i8>
  %40 = zext <8 x i8> %39 to <8 x i16>
  %41 = sub <8 x i16> zeroinitializer, %40
  %42 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %36) nounwind
  %43 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %41) nounwind
  %44 = mul <8 x i16> %42, %42
  %45 = mul <8 x i16> %43, %43
  %46 = add <8 x i16> %45, %44
  %47 = tail call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %31, <8 x i16> %46) nounwind
  %48 = bitcast <8 x i16> %47 to <2 x i64>
  %49 = shufflevector <2 x i64> %48, <2 x i64> undef, <1 x i32> zeroinitializer
  %50 = bitcast <1 x i64> %49 to <4 x i16>
  %51 = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %50, <4 x i16> undef) nounwind
  %52 = tail call <4 x i16> @llvm.arm.neon.vqshiftnu.v4i16(<4 x i32> %51, <4 x i32> <i32 -6, i32 -6, i32 -6, i32 -6>)
  %53 = bitcast <4 x i16> %52 to <1 x i64>
  %54 = shufflevector <1 x i64> %53, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %55 = bitcast <2 x i64> %54 to <8 x i16>
  %56 = tail call <8 x i8> @llvm.arm.neon.vshiftn.v8i8(<8 x i16> %55, <8 x i16> <i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8>)
  %57 = shufflevector <2 x i64> %20, <2 x i64> undef, <1 x i32> <i32 1>
  %58 = bitcast <1 x i64> %57 to <8 x i8>
  %59 = zext <8 x i8> %58 to <8 x i16>
  %60 = sub <8 x i16> zeroinitializer, %59
  %61 = shufflevector <2 x i64> %21, <2 x i64> undef, <1 x i32> <i32 1>
  %62 = bitcast <1 x i64> %61 to <8 x i8>
  %63 = zext <8 x i8> %62 to <8 x i16>
  %64 = sub <8 x i16> %63, zeroinitializer
  %65 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %60) nounwind
  %66 = mul <8 x i16> %65, %65
  %67 = add <8 x i16> zeroinitializer, %66
  %68 = shufflevector <2 x i64> %27, <2 x i64> undef, <1 x i32> <i32 1>
  %69 = bitcast <1 x i64> %68 to <8 x i8>
  %70 = zext <8 x i8> %69 to <8 x i16>
  %71 = sub <8 x i16> zeroinitializer, %70
  %72 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> undef) nounwind
  %73 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %71) nounwind
  %74 = mul <8 x i16> %72, %72
  %75 = mul <8 x i16> %73, %73
  %76 = add <8 x i16> %75, %74
  %77 = shufflevector <2 x i64> %32, <2 x i64> undef, <1 x i32> <i32 1>
  %78 = bitcast <1 x i64> %77 to <8 x i8>
  %79 = zext <8 x i8> %78 to <8 x i16>
  %80 = sub <8 x i16> %79, zeroinitializer
  %81 = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %80) nounwind
  %82 = mul <8 x i16> %81, %81
  %83 = add <8 x i16> zeroinitializer, %82
  %84 = tail call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %76, <8 x i16> %83) nounwind
  %85 = tail call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %67, <8 x i16> %84) nounwind
  %86 = bitcast <8 x i16> %85 to <2 x i64>
  %87 = shufflevector <2 x i64> %86, <2 x i64> undef, <1 x i32> <i32 1>
  %88 = bitcast <1 x i64> %87 to <4 x i16>
  %89 = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %88, <4 x i16> undef) nounwind
  %90 = tail call <4 x i16> @llvm.arm.neon.vqrshiftnu.v4i16(<4 x i32> %89, <4 x i32> <i32 -6, i32 -6, i32 -6, i32 -6>)
  %91 = bitcast <4 x i16> %90 to <1 x i64>
  %92 = shufflevector <1 x i64> undef, <1 x i64> %91, <2 x i32> <i32 0, i32 1>
  %93 = bitcast <2 x i64> %92 to <8 x i16>
  %94 = tail call <8 x i8> @llvm.arm.neon.vshiftn.v8i8(<8 x i16> %93, <8 x i16> <i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8>)
  %95 = bitcast <8 x i8> %56 to <1 x i64>
  %96 = bitcast <8 x i8> %94 to <1 x i64>
  %97 = shufflevector <1 x i64> %95, <1 x i64> %96, <2 x i32> <i32 0, i32 1>
  %98 = bitcast <2 x i64> %97 to <16 x i8>
  tail call void @llvm.arm.neon.vst1.p0.v16i8(ptr null, <16 x i8> %98, i32 1)
  %99 = icmp slt i32 undef, undef
  br i1 %99, label %5, label %3
}

declare <4 x i16> @llvm.arm.neon.vqshiftnu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone

declare <8 x i8> @llvm.arm.neon.vshiftn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone

declare <4 x i16> @llvm.arm.neon.vqrshiftnu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16>) nounwind readnone
