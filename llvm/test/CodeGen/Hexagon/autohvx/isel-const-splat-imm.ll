; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; Check that this doesn't crash. A "splat_vector" was causing trouble,
; initially, so check that a vsplat appears in the output.
; CHECK: vsplat

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = external dllexport local_unnamed_addr global i32 (i32, i32, i8*)*, align 4

; Function Attrs: noinline
define dso_local fastcc void @f0(i8* %a0, i32 %a1) unnamed_addr #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v0 = add nsw <8 x i32> zeroinitializer, <i32 -22, i32 -22, i32 -22, i32 -22, i32 -22, i32 -22, i32 -22, i32 -22>
  %v1 = load <8 x i32>, <8 x i32>* undef, align 32
  %v2 = shl <8 x i32> %v1, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v3 = ashr exact <8 x i32> %v2, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v4 = add nsw <8 x i32> %v3, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v5 = zext <8 x i32> %v4 to <8 x i64>
  %v6 = mul nuw nsw <8 x i64> %v5, <i64 411541167360, i64 411541167360, i64 411541167360, i64 411541167360, i64 411541167360, i64 411541167360, i64 411541167360, i64 411541167360>
  %v7 = add nuw nsw <8 x i64> %v6, <i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824>
  %v8 = lshr <8 x i64> %v7, <i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31>
  %v9 = trunc <8 x i64> %v8 to <8 x i32>
  %v10 = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v9, <8 x i32> <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>)
  %v11 = shl nuw <8 x i32> %v10, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v12 = ashr exact <8 x i32> %v11, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v13 = xor <8 x i32> %v12, <i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128>
  %v14 = add nsw <8 x i32> %v13, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v15 = sitofp <8 x i32> %v14 to <8 x float>
  %v16 = fmul nnan nsz <8 x float> %v15, <float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000, float 0x3F14E70560000000>
  %v17 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %v16)
  %v18 = fdiv <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %v17
  %v19 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %v18, <8 x float> <float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000, float 0x4026469520000000>, <8 x float> <float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02>)
  %v20 = fcmp olt <8 x float> %v19, <float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02>
  %v21 = select <8 x i1> %v20, <8 x float> %v19, <8 x float> <float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02, float 1.270000e+02>
  %v22 = fcmp ogt <8 x float> %v21, <float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02>
  %v23 = select <8 x i1> %v22, <8 x float> %v21, <8 x float> <float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02, float -1.280000e+02>
  %v24 = call <8 x float> @llvm.round.v8f32(<8 x float> %v23)
  %v25 = fptosi <8 x float> %v24 to <8 x i8>
  %v26 = sext <8 x i8> %v25 to <8 x i32>
  %v27 = add nsw <8 x i32> %v26, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v28 = load <8 x i32>, <8 x i32>* undef, align 16
  %v29 = mul nsw <8 x i32> %v27, %v28
  %v30 = sext <8 x i32> %v29 to <8 x i64>
  %v31 = mul nsw <8 x i64> %v30, <i64 1077952632, i64 1077952632, i64 1077952632, i64 1077952632, i64 1077952632, i64 1077952632, i64 1077952632, i64 1077952632>
  %v32 = add nsw <8 x i64> %v31, <i64 137438953472, i64 137438953472, i64 137438953472, i64 137438953472, i64 137438953472, i64 137438953472, i64 137438953472, i64 137438953472>
  %v33 = ashr <8 x i64> %v32, <i64 38, i64 38, i64 38, i64 38, i64 38, i64 38, i64 38, i64 38>
  %v34 = trunc <8 x i64> %v33 to <8 x i32>
  %v35 = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v34, <8 x i32> <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>)
  %v36 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %v35, <8 x i32> zeroinitializer)
  %v37 = shl nuw <8 x i32> %v36, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v38 = ashr exact <8 x i32> %v37, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v39 = xor <8 x i32> %v38, <i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128>
  %v40 = add nsw <8 x i32> %v39, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v41 = mul nsw <8 x i32> %v40, %v0
  %v42 = sext <8 x i32> %v41 to <8 x i64>
  %v43 = mul nsw <8 x i64> %v42, <i64 1268103552, i64 1268103552, i64 1268103552, i64 1268103552, i64 1268103552, i64 1268103552, i64 1268103552, i64 1268103552>
  %v44 = add nsw <8 x i64> %v43, <i64 34359738368, i64 34359738368, i64 34359738368, i64 34359738368, i64 34359738368, i64 34359738368, i64 34359738368, i64 34359738368>
  %v45 = ashr <8 x i64> %v44, <i64 36, i64 36, i64 36, i64 36, i64 36, i64 36, i64 36, i64 36>
  %v46 = trunc <8 x i64> %v45 to <8 x i32>
  %v47 = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v46, <8 x i32> <i32 64, i32 64, i32 64, i32 64, i32 64, i32 64, i32 64, i32 64>)
  %v48 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %v47, <8 x i32> <i32 -191, i32 -191, i32 -191, i32 -191, i32 -191, i32 -191, i32 -191, i32 -191>)
  %v49 = shl <8 x i32> %v48, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v50 = add <8 x i32> %v49, <i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608>
  %v51 = ashr exact <8 x i32> %v50, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %v52 = add nsw <8 x i32> %v51, <i32 -63, i32 -63, i32 -63, i32 -63, i32 -63, i32 -63, i32 -63, i32 -63>
  %v53 = sext <8 x i32> %v52 to <8 x i64>
  %v54 = mul nsw <8 x i64> %v53, <i64 2240554232, i64 2240554232, i64 2240554232, i64 2240554232, i64 2240554232, i64 2240554232, i64 2240554232, i64 2240554232>
  %v55 = add nsw <8 x i64> %v54, <i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824, i64 1073741824>
  %v56 = lshr <8 x i64> %v55, <i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31>
  %v57 = trunc <8 x i64> %v56 to <8 x i32>
  %v58 = add nsw <8 x i32> zeroinitializer, %v57
  %v59 = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v58, <8 x i32> <i32 69, i32 69, i32 69, i32 69, i32 69, i32 69, i32 69, i32 69>)
  %v60 = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %v59, <8 x i32> <i32 -186, i32 -186, i32 -186, i32 -186, i32 -186, i32 -186, i32 -186, i32 -186>)
  %v61 = trunc <8 x i32> %v60 to <8 x i8>
  %v62 = add <8 x i8> %v61, <i8 58, i8 58, i8 58, i8 58, i8 58, i8 58, i8 58, i8 58>
  %v63 = getelementptr inbounds i8, i8* %a0, i32 undef
  %v64 = bitcast i8* %v63 to <8 x i8>*
  store <8 x i8> %v62, <8 x i8>* %v64, align 8
  %v65 = load i32 (i32, i32, i8*)*, i32 (i32, i32, i8*)** @g0, align 4
  %v66 = tail call i32 %v65(i32 14, i32 %a1, i8* nonnull undef)
  unreachable

b2:                                               ; preds = %b0
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <8 x i32> @llvm.smin.v8i32(<8 x i32>, <8 x i32>) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <8 x float> @llvm.round.v8f32(<8 x float>) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <8 x i32> @llvm.smax.v8i32(<8 x i32>, <8 x i32>) #1

attributes #0 = { noinline "target-cpu"="hexagonv69" "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
