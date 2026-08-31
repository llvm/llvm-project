; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; Register allocation packs the per-lane index/address computation of these four
; float4 loads into overlapping address registers, a false WAR dependency that
; otherwise pins the loads apart. The pass renames the address chains so the
; post-RA load-cluster scheduler can issue all four loads as a burst.

; CHECK-LABEL: _Z12uavReadSpeedI15HIP_vector_typeIfLj4EEEvPT_S3_PKjjS2_:
; CHECK: global_load_dwordx4
; CHECK-NEXT: global_load_dwordx4
; CHECK-NEXT: global_load_dwordx4
; CHECK: global_load_dwordx4

source_filename = "repro_hipperf_uavreadspeed.cpp"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.HIP_vector_type = type { %struct.HIP_vector_base }
%struct.HIP_vector_base = type { float, float, float, float }

$_Z12uavReadSpeedI15HIP_vector_typeIfLj4EEEvPT_S3_PKjjS2_ = comdat any

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define protected amdgpu_kernel void @_Z12uavReadSpeedI15HIP_vector_typeIfLj4EEEvPT_S3_PKjjS2_(ptr addrspace(1) noalias nofree noundef readonly captures(none) %arg, ptr addrspace(1) noalias nofree noundef writeonly captures(none) %arg1, ptr addrspace(1) noalias nofree noundef readonly captures(none) %arg2, i32 noundef %arg3, ptr addrspace(4) nofree noundef readnone byref(%struct.HIP_vector_type) align 16 captures(none) %arg4) local_unnamed_addr #0 comdat {
bb:
  %i = tail call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i5 = getelementptr inbounds nuw i8, ptr addrspace(4) %i, i64 12
  %i6 = load i16, ptr addrspace(4) %i5, align 4, !range !16, !invariant.load !17, !noundef !17
  %i7 = zext nneg i16 %i6 to i32
  %i8 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %i9 = mul i32 %i8, %i7
  %i10 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %i11 = add i32 %i9, %i10
  %i12 = icmp eq i32 %arg3, 1
  br i1 %i12, label %bb13, label %bb20

bb13:                                             ; preds = %bb
  %i14 = load i32, ptr addrspace(1) %arg2, align 4, !tbaa !18
  %i15 = urem i32 %i11, %i14
  %i16 = zext i32 %i15 to i64
  %i17 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %arg, i64 %i16
  %i18 = sext i32 %i11 to i64
  %i19 = getelementptr inbounds [16 x i8], ptr addrspace(1) %arg1, i64 %i18
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %i19, ptr addrspace(1) noundef align 16 dereferenceable(16) %i17, i64 16, i1 false)
  br label %bb81

bb20:                                             ; preds = %bb
  %i21 = lshr i32 %arg3, 2
  %i22 = icmp eq i32 %i21, 0
  br i1 %i22, label %bb42, label %bb23

bb23:                                             ; preds = %bb20
  %i24 = load i32, ptr addrspace(1) %arg2, align 4, !tbaa !18
  %i25 = urem i32 %i11, %i24
  %i26 = getelementptr inbounds nuw i8, ptr addrspace(1) %arg2, i64 12
  %i27 = load <2 x i32>, ptr addrspace(1) %i26, align 4, !tbaa !18
  %i28 = insertelement <2 x i32> poison, i32 %i25, i64 0
  %i29 = shufflevector <2 x i32> %i28, <2 x i32> poison, <2 x i32> zeroinitializer
  %i30 = add <2 x i32> %i27, %i29
  %i31 = getelementptr inbounds nuw i8, ptr addrspace(1) %arg2, i64 4
  %i32 = load <2 x i32>, ptr addrspace(1) %i31, align 4, !tbaa !18
  %i33 = add <2 x i32> %i32, %i29
  %i34 = getelementptr inbounds nuw i8, ptr addrspace(1) %arg2, i64 20
  %i35 = load i32, ptr addrspace(1) %i34, align 4, !tbaa !18
  %i36 = insertelement <2 x i32> poison, i32 %i35, i64 0
  %i37 = shufflevector <2 x i32> %i36, <2 x i32> poison, <2 x i32> zeroinitializer
  br label %bb49

bb38:                                             ; preds = %bb49
  %i39 = fadd contract <4 x float> %i61, %i66
  %i40 = fadd contract <4 x float> %i71, %i39
  %i41 = fadd contract <4 x float> %i76, %i40
  br label %bb42

bb42:                                             ; preds = %bb38, %bb20
  %i43 = phi <4 x float> [ zeroinitializer, %bb20 ], [ %i41, %bb38 ]
  %i44 = sext i32 %i11 to i64
  %i45 = getelementptr inbounds [16 x i8], ptr addrspace(1) %arg1, i64 %i44
  %i46 = shufflevector <4 x float> %i43, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  store <2 x float> %i46, ptr addrspace(1) %i45, align 16
  %i47 = getelementptr inbounds nuw i8, ptr addrspace(1) %i45, i64 8
  %i48 = shufflevector <4 x float> %i43, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  store <2 x float> %i48, ptr addrspace(1) %i47, align 8
  br label %bb81

bb49:                                             ; preds = %bb49, %bb23
  %i50 = phi i32 [ 0, %bb23 ], [ %i79, %bb49 ]
  %i51 = phi <4 x float> [ zeroinitializer, %bb23 ], [ %i76, %bb49 ]
  %i52 = phi <4 x float> [ zeroinitializer, %bb23 ], [ %i71, %bb49 ]
  %i53 = phi <4 x float> [ zeroinitializer, %bb23 ], [ %i66, %bb49 ]
  %i54 = phi <4 x float> [ zeroinitializer, %bb23 ], [ %i61, %bb49 ]
  %i55 = phi <2 x i32> [ %i33, %bb23 ], [ %i77, %bb49 ]
  %i56 = phi <2 x i32> [ %i30, %bb23 ], [ %i78, %bb49 ]
  %i57 = extractelement <2 x i32> %i55, i64 0
  %i58 = zext i32 %i57 to i64
  %i59 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %arg, i64 %i58
  %i60 = load <4 x float>, ptr addrspace(1) %i59, align 16, !tbaa !19
  %i61 = fadd contract <4 x float> %i54, %i60
  %i62 = extractelement <2 x i32> %i55, i64 1
  %i63 = zext i32 %i62 to i64
  %i64 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %arg, i64 %i63
  %i65 = load <4 x float>, ptr addrspace(1) %i64, align 16, !tbaa !19
  %i66 = fadd contract <4 x float> %i53, %i65
  %i67 = extractelement <2 x i32> %i56, i64 0
  %i68 = zext i32 %i67 to i64
  %i69 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %arg, i64 %i68
  %i70 = load <4 x float>, ptr addrspace(1) %i69, align 16, !tbaa !19
  %i71 = fadd contract <4 x float> %i52, %i70
  %i72 = extractelement <2 x i32> %i56, i64 1
  %i73 = zext i32 %i72 to i64
  %i74 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %arg, i64 %i73
  %i75 = load <4 x float>, ptr addrspace(1) %i74, align 16, !tbaa !19
  %i76 = fadd contract <4 x float> %i51, %i75
  %i77 = add <2 x i32> %i37, %i55
  %i78 = add <2 x i32> %i37, %i56
  %i79 = add nuw nsw i32 %i50, 1
  %i80 = icmp eq i32 %i79, %i21
  br i1 %i80, label %bb38, label %bb49, !llvm.loop !20

bb81:                                             ; preds = %bb42, %bb13
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5, !10}
!opencl.ocl.version = !{!15}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git 657cfa16903ad4c5921a6f8992bb50009da0256f)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !12, i64 0}
!11 = !{!"__libc_errno", !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !{i32 2, i32 0}
!16 = !{i16 1, i16 1025}
!17 = !{}
!18 = !{!7, !7, i64 0}
!19 = !{!8, !8, i64 0}
!20 = distinct !{!20, !21}
!21 = !{!"llvm.loop.mustprogress"}
