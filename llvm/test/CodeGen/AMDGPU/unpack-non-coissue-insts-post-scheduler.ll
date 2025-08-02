; TODO: change variable names. Make test smaller if possible

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float)

; Function Attrs: nofree norecurse nounwind
define amdgpu_kernel void @attn_fwd(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg readonly captures(none) %1, ptr addrspace(1) inreg readonly captures(none) %2, ptr addrspace(1) inreg writeonly captures(none) %3, ptr addrspace(1) inreg writeonly captures(none) %4, i32 inreg %5, i32 inreg %6, i32 inreg %7, i32 inreg %8, i32 inreg %9, i32 inreg %10, i32 inreg %11, i32 inreg %12, i32 inreg %13, i32 inreg %14, i32 inreg %15, i32 inreg %16, i32 inreg %17, i32 inreg %18, i32 inreg %19, i32 inreg %20, i32 inreg %21, i32 inreg %22, float inreg %23, i32 inreg %24, ptr addrspace(1) inreg readnone captures(none) %25, i32 inreg %26, ptr addrspace(1) inreg readnone captures(none) %27) local_unnamed_addr {
  %29 = tail call i32 @llvm.amdgcn.workgroup.id.x()
    
  %96 = sext i32 %8 to i64
  %97 = getelementptr half, ptr addrspace(1) %1, i64 %96
  
  %115 = icmp slt i32 %29, 16384

  %135 = icmp slt i32 %29, 1
  
  %215 = getelementptr half, ptr addrspace(3) @global_smem, i32 %29
  %216 = load <8 x half>, ptr addrspace(3) %215, align 16
  
  %276 = shl nuw nsw i32 %29, 7
  
  %396 = getelementptr half, ptr addrspace(1) %97, i64 1
  %397 = sext i32 %13 to i64
  %398 = getelementptr half, ptr addrspace(1) %97, i64 %397
  
  %536 = fsub float 0xFFF0000000000000, 0.5
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536)
  
  %538 = getelementptr half, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16384), i32 %29
  %539 = load <8 x half>, ptr addrspace(3) %538, align 16
  
  %573 = icmp ult i32 1, 511
  br i1 %573, label %575, label %574

574:                                              ; preds = %28
  br label %575

575:                                              ; preds = %574, %28
  %610 = shufflevector <8 x half> %539, <8 x half> poison, <2 x i32> <i32 0, i32 1>
  
  br label %686

686:                                              ; preds = %575, %686
  %.pn347561 = phi float [ %537, %575 ], [ %1329, %686 ]
  
  
  %690 = phi i32 [ 0, %575 ], [ %1120, %686 ]
  %691 = phi ptr addrspace(1) [ %398, %575 ], [ %1117, %686 ]
  %692 = phi ptr addrspace(1) [ %396, %575 ], [ %1116, %686 ]
  
  %695 = phi <2 x half> [ %610, %575 ], [ %1414, %686 ]
  
  
  %759 = phi <2 x float> [ zeroinitializer, %575 ], [ %1478, %686 ]
  %760 = phi <2 x float> [ zeroinitializer, %575 ], [ %1478, %686 ]

  %tmp6 = phi <2 x float> [ zeroinitializer, %575 ], [ %tmp5, %686 ]
  %tmp7 = phi <2 x float> [ zeroinitializer, %575 ], [ %tmp5, %686 ]
  
  %871 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %216, <8 x half> %216, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %872 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %216, <8 x half> %216, <16 x float> %871, i32 0, i32 0, i32 0)
  %879 = extractelement <16 x float> %872, i64 0
  
  
  %957 = insertelement <2 x float> poison, float %.pn347561, i64 0
  %958 = shufflevector <2 x float> %957, <2 x float> poison, <2 x i32> zeroinitializer
  %959 = fmul <2 x float> %759, %958
  %960 = fmul <2 x float> %760, %958
  
  %tmp1 = fmul <2 x float> %tmp6, %958
  %tmp2 = fmul <2 x float> %tmp7, %958  
  
  %1048 = shufflevector <2 x half> %695, <2 x half> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  
  %1116 = getelementptr half, ptr addrspace(1) %692, i64 1
  %1117 = getelementptr half, ptr addrspace(1) %691, i64 %397
  
  %1119 = icmp slt i32 %690, 2
  %1120 = select i1 %1119, i32 %690, i32 0
  %.idx359 = shl i32 %1120, 14
  %1121 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %.idx359
  
  %1140 = shufflevector <8 x half> %1048, <8 x half> %1048, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  
  %1157 = shufflevector <2 x float> %959, <2 x float> %960, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>

  %1173 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %1048, <8 x half> %1140, <16 x float> %1157, i32 0, i32 0, i32 0)
  %tmp4 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %1048, <8 x half> %1140, <16 x float> %tmp3, i32 0, i32 0, i32 0)
  
  
  %1329 = tail call float @llvm.amdgcn.exp2.f32(float %879)
  
  %.idx367 = shl i32 %690, 14
  %1404 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 32768), i32 %.idx367
  
  %1412 = add nuw nsw i32 0, 64
  %1413 = icmp samesign ult i32 0, 7936
  %1414 = shufflevector <8 x half> %1140, <8 x half> poison, <2 x i32> <i32 0, i32 1>
  
  %1478 = shufflevector <16 x float> %1173, <16 x float> poison, <2 x i32> <i32 0, i32 1>
  %tmp5 = shufflevector <16 x float> %tmp4, <16 x float> poison, <2 x i32> <i32 0, i32 1>
  
  br i1 %1413, label %686, label %1510

1510:                                             ; preds = %686
  ret void
}
