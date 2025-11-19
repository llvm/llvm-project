; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

; FIXME: Scale operands of WMMA are limited to low 256 VGPRs
;        currently we are spilling it because all low VGPRs are occupied even though our budget is higher.
; Make sure we do not spill scale operands because of the low 256 restriction.
; CHECK: ; ScratchSize: 12
; CHECK: ; Occupancy: 1

define amdgpu_kernel void @spill_scale_test(float %arg, float %arg1, float %arg2, float %arg3, float %arg4, float %arg5, float %arg6, float %arg7, <16 x i32> %arg8, float %arg9, <16 x i32> %arg10, float %arg11, <16 x i8> %arg12) #0 {
bb:
  %i = shufflevector <16 x i8> %arg12, <16 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  tail call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) null, ptr addrspace(3) null, i32 0, i32 0)
  %i13 = bitcast <64 x i8> %i to <16 x i32>
  tail call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) null, ptr addrspace(3) null, i32 0, i32 0)
  %i14 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) null)
  %i15 = bitcast <2 x i32> %i14 to <8 x i8>
  %i16 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) null)
  %i17 = shufflevector <8 x i8> %i15, <8 x i8> zeroinitializer, <64 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i18 = shufflevector <64 x i8> zeroinitializer, <64 x i8> %i17, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 64, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i19 = insertelement <64 x i8> %i18, i8 0, i64 57
  %i20 = bitcast <64 x i8> %i19 to <16 x i32>
  %.extract2214 = extractelement <2 x i32> %i16, i64 0
  %i21 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %i20, i32 0, <16 x i32> %i13, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i22 = extractelement <8 x float> %i21, i64 0
  %i23 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> zeroinitializer, i32 0, <16 x i32> zeroinitializer, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 %.extract2214, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i24 = extractelement <8 x float> %i23, i64 0
  %i25 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %arg8, i32 0, <16 x i32> zeroinitializer, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i26 = extractelement <8 x float> %i25, i64 0
  %i27 = fmul float %i22, 0.000000e+00
  %i28 = fmul float %i24, 0.000000e+00
  %i29 = insertelement <2 x float> zeroinitializer, float %i26, i64 1
  %i30 = insertelement <2 x float> zeroinitializer, float %i28, i64 0
  %i31 = insertelement <2 x float> zeroinitializer, float %arg11, i64 0
  %i32 = fadd <2 x float> %i31, %i30
  %i33 = insertelement <2 x float> zeroinitializer, float %arg9, i64 0
  %i34 = fadd <2 x float> %i33, %i32
  %i35 = insertelement <2 x float> zeroinitializer, float %arg7, i64 0
  %i36 = fadd <2 x float> %i35, %i34
  %i37 = insertelement <2 x float> zeroinitializer, float %arg1, i64 0
  %i38 = fadd <2 x float> %i37, %i36
  %i39 = insertelement <2 x float> zeroinitializer, float %arg6, i64 0
  %i40 = fadd <2 x float> %i39, %i38
  %i41 = insertelement <2 x float> zeroinitializer, float %arg4, i64 0
  %i42 = fadd <2 x float> %i41, %i40
  %i43 = insertelement <2 x float> zeroinitializer, float %arg5, i64 0
  %i44 = fadd <2 x float> %i43, %i42
  %i45 = insertelement <2 x float> zeroinitializer, float %arg3, i64 0
  %i46 = fadd <2 x float> %i45, %i44
  %i47 = insertelement <2 x float> zeroinitializer, float %arg, i64 0
  %i48 = insertelement <2 x float> zeroinitializer, float %arg2, i64 0
  %i49 = fadd <2 x float> %i48, %i46
  %i50 = fadd <2 x float> %i29, %i49
  %i51 = fadd <2 x float> %i47, %i50
  %i52 = insertelement <8 x float> zeroinitializer, float %i27, i64 0
  %i53 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> %i52, float 0.000000e+00)
  %i54 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> splat (float 0x7FF8000000000000), float 0.000000e+00)
  %i55 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> splat (float 1.000000e+00), float 0.000000e+00)
  %.extract1415 = extractelement <2 x i32> %i53, i64 0
  %.extract1416 = extractelement <2 x i32> %i54, i64 0
  %.extract1424 = extractelement <2 x i32> %i55, i64 0
  %i56 = tail call i32 @llvm.amdgcn.ds.swizzle(i32 %.extract1416, i32 0)
  %i57 = bitcast i32 %.extract1415 to <4 x i8>
  %i58 = shufflevector <4 x i8> %i57, <4 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i59 = bitcast i32 %i56 to <4 x i8>
  %i60 = bitcast i32 %.extract1424 to <4 x i8>
  %i61 = shufflevector <4 x i8> %i60, <4 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i62 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) null)
  %i63 = bitcast <2 x i32> %i62 to <8 x i8>
  %i64 = shufflevector <8 x i8> %i63, <8 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i65 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) null)
  %i66 = bitcast <2 x i32> %i65 to <8 x i8>
  %i67 = shufflevector <8 x i8> %i66, <8 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i68 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) null)
  %i69 = bitcast <2 x i32> %i68 to <8 x i8>
  %i70 = shufflevector <8 x i8> %i69, <8 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i71 = tail call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) getelementptr (i8, ptr addrspace(3) null, i32 75232))
  %i72 = shufflevector <64 x i8> zeroinitializer, <64 x i8> %i58, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 64, i32 65, i32 66, i32 67>
  %i73 = bitcast <64 x i8> %i72 to <16 x i32>
  %i74 = shufflevector <4 x i8> %i59, <4 x i8> zeroinitializer, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i75 = shufflevector <64 x i8> %i74, <64 x i8> %i61, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 64, i32 65, i32 66, i32 67, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i76 = bitcast <64 x i8> %i75 to <16 x i32>
  %i77 = shufflevector <64 x i8> zeroinitializer, <64 x i8> %i64, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %i78 = bitcast <64 x i8> %i77 to <16 x i32>
  %i79 = bitcast <64 x i8> %i67 to <16 x i32>
  %i80 = shufflevector <64 x i8> zeroinitializer, <64 x i8> %i70, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71>
  %i81 = bitcast <64 x i8> %i80 to <16 x i32>
  %.extract1434 = extractelement <2 x i32> %i71, i64 0
  %i82 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %i78, i32 0, <16 x i32> zeroinitializer, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i83 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %arg10, i32 0, <16 x i32> %i73, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i84 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %i79, i32 0, <16 x i32> %i73, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2139062143, i1 false, i1 false)
  %i85 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %i81, i32 0, <16 x i32> %arg8, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2139062143, i1 false, i1 false)
  %i86 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> splat (i32 16843009), i32 0, <16 x i32> %arg10, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i87 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> zeroinitializer, i32 0, <16 x i32> zeroinitializer, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i88 = tail call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> splat (i32 1), i32 0, <16 x i32> %i76, i16 0, <8 x float> zeroinitializer, i32 0, i32 0, i32 %.extract1434, i32 0, i32 0, i32 0, i1 false, i1 false)
  %i89 = fdiv <8 x float> %i82, zeroinitializer
  %i90 = fcmp uno <8 x float> %i89, zeroinitializer
  %i91 = select <8 x i1> %i90, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i92 = bitcast <8 x bfloat> %i91 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i92, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i93 = fdiv <8 x float> %i83, zeroinitializer
  %i94 = fcmp uno <8 x float> %i93, zeroinitializer
  %i95 = select <8 x i1> %i94, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i96 = bitcast <8 x bfloat> %i95 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i96, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i97 = fcmp uno <8 x float> %i84, zeroinitializer
  %i98 = select <8 x i1> %i97, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i99 = bitcast <8 x bfloat> %i98 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i99, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i100 = fcmp uno <8 x float> %i85, zeroinitializer
  %i101 = select <8 x i1> %i100, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i102 = bitcast <8 x bfloat> %i101 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i102, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i103 = fcmp uno <8 x float> %i86, zeroinitializer
  %i104 = select <8 x i1> %i103, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i105 = bitcast <8 x bfloat> %i104 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i105, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i106 = fcmp uno <8 x float> %i87, zeroinitializer
  %i107 = select <8 x i1> %i106, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i108 = bitcast <8 x bfloat> %i107 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i108, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i109 = shufflevector <2 x float> %i51, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  %i110 = shufflevector <4 x float> %i109, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %i111 = fmul <8 x float> %i88, %i110
  %i112 = fcmp uno <8 x float> %i111, zeroinitializer
  %i113 = select <8 x i1> %i112, <8 x bfloat> splat (bfloat 0xR3F80), <8 x bfloat> zeroinitializer
  %i114 = bitcast <8 x bfloat> %i113 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %i114, ptr addrspace(8) null, i32 0, i32 0, i32 0)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,128" "amdgpu-waves-per-eu"="1,1" }
