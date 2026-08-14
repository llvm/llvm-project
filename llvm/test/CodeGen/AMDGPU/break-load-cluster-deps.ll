; RUN: llc -mcpu=gfx90a < %s | FileCheck %s
; CHECK-LABEL: This Inner Loop
; CHECK: global_load_dwordx4
; CHECK-NEXT: global_load_dwordx4
; CHECK-NEXT: add
; CHECK-NEXT: global_load_dwordx4
; CHECK-NEXT: add
; CHECK-NEXT: global_load_dwordx4
; CHECK-LABEL: This Inner Loop
; CHECK: global_load_dwordx4
; CHECK-NEXT: global_load_dwordx4
; CHECK-NEXT: add
; CHECK-NEXT: global_load_dwordx4
; CHECK-NEXT: add
; CHECK-NEXT: global_load_dwordx4
; ModuleID = 'repro_hipperf_uavreadspeed.cpp'
source_filename = "repro_hipperf_uavreadspeed.cpp"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.HIP_vector_type = type { %struct.HIP_vector_base }
%struct.HIP_vector_base = type { float, float, float, float }
%struct.HIP_vector_type.0 = type { %struct.HIP_vector_base.1 }
%struct.HIP_vector_base.1 = type { double, double }

$_Z12uavReadSpeedI15HIP_vector_typeIfLj4EEEvPT_S3_PKjjS2_ = comdat any

$_Z12uavReadSpeedI15HIP_vector_typeIdLj2EEEvPT_S3_PKjjS2_ = comdat any

@__hip_cuid_6dcd12a76005a3fa = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_6dcd12a76005a3fa to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define protected amdgpu_kernel void @_Z12uavReadSpeedI15HIP_vector_typeIfLj4EEEvPT_S3_PKjjS2_(ptr addrspace(1) noalias nofree noundef readonly captures(none) %0, ptr addrspace(1) noalias nofree noundef writeonly captures(none) %1, ptr addrspace(1) noalias nofree noundef readonly captures(none) %2, i32 noundef %3, ptr addrspace(4) nofree noundef readnone byref(%struct.HIP_vector_type) align 16 captures(none) %4) local_unnamed_addr #0 comdat {
  %6 = tail call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !range !16, !invariant.load !17, !noundef !17
  %9 = zext nneg i16 %8 to i32
  %10 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %11 = mul i32 %10, %9
  %12 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %13 = add i32 %11, %12
  %14 = icmp eq i32 %3, 1
  br i1 %14, label %15, label %22

15:                                               ; preds = %5
  %16 = load i32, ptr addrspace(1) %2, align 4, !tbaa !18
  %17 = urem i32 %13, %16
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %18
  %20 = sext i32 %13 to i64
  %21 = getelementptr inbounds [16 x i8], ptr addrspace(1) %1, i64 %20
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %21, ptr addrspace(1) noundef align 16 dereferenceable(16) %19, i64 16, i1 false)
  br label %83

22:                                               ; preds = %5
  %23 = lshr i32 %3, 2
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %44, label %25

25:                                               ; preds = %22
  %26 = load i32, ptr addrspace(1) %2, align 4, !tbaa !18
  %27 = urem i32 %13, %26
  %28 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 12
  %29 = load <2 x i32>, ptr addrspace(1) %28, align 4, !tbaa !18
  %30 = insertelement <2 x i32> poison, i32 %27, i64 0
  %31 = shufflevector <2 x i32> %30, <2 x i32> poison, <2 x i32> zeroinitializer
  %32 = add <2 x i32> %29, %31
  %33 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 4
  %34 = load <2 x i32>, ptr addrspace(1) %33, align 4, !tbaa !18
  %35 = add <2 x i32> %34, %31
  %36 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 20
  %37 = load i32, ptr addrspace(1) %36, align 4, !tbaa !18
  %38 = insertelement <2 x i32> poison, i32 %37, i64 0
  %39 = shufflevector <2 x i32> %38, <2 x i32> poison, <2 x i32> zeroinitializer
  br label %51

40:                                               ; preds = %51
  %41 = fadd contract <4 x float> %63, %68
  %42 = fadd contract <4 x float> %73, %41
  %43 = fadd contract <4 x float> %78, %42
  br label %44

44:                                               ; preds = %40, %22
  %45 = phi <4 x float> [ zeroinitializer, %22 ], [ %43, %40 ]
  %46 = sext i32 %13 to i64
  %47 = getelementptr inbounds [16 x i8], ptr addrspace(1) %1, i64 %46
  %48 = shufflevector <4 x float> %45, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  store <2 x float> %48, ptr addrspace(1) %47, align 16
  %49 = getelementptr inbounds nuw i8, ptr addrspace(1) %47, i64 8
  %50 = shufflevector <4 x float> %45, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  store <2 x float> %50, ptr addrspace(1) %49, align 8
  br label %83

51:                                               ; preds = %25, %51
  %52 = phi i32 [ 0, %25 ], [ %81, %51 ]
  %53 = phi <4 x float> [ zeroinitializer, %25 ], [ %78, %51 ]
  %54 = phi <4 x float> [ zeroinitializer, %25 ], [ %73, %51 ]
  %55 = phi <4 x float> [ zeroinitializer, %25 ], [ %68, %51 ]
  %56 = phi <4 x float> [ zeroinitializer, %25 ], [ %63, %51 ]
  %57 = phi <2 x i32> [ %35, %25 ], [ %79, %51 ]
  %58 = phi <2 x i32> [ %32, %25 ], [ %80, %51 ]
  %59 = extractelement <2 x i32> %57, i64 0
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %60
  %62 = load <4 x float>, ptr addrspace(1) %61, align 16, !tbaa !19
  %63 = fadd contract <4 x float> %56, %62
  %64 = extractelement <2 x i32> %57, i64 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %65
  %67 = load <4 x float>, ptr addrspace(1) %66, align 16, !tbaa !19
  %68 = fadd contract <4 x float> %55, %67
  %69 = extractelement <2 x i32> %58, i64 0
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %70
  %72 = load <4 x float>, ptr addrspace(1) %71, align 16, !tbaa !19
  %73 = fadd contract <4 x float> %54, %72
  %74 = extractelement <2 x i32> %58, i64 1
  %75 = zext i32 %74 to i64
  %76 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %75
  %77 = load <4 x float>, ptr addrspace(1) %76, align 16, !tbaa !19
  %78 = fadd contract <4 x float> %53, %77
  %79 = add <2 x i32> %39, %57
  %80 = add <2 x i32> %39, %58
  %81 = add nuw nsw i32 %52, 1
  %82 = icmp eq i32 %81, %23
  br i1 %82, label %40, label %51, !llvm.loop !20

83:                                               ; preds = %44, %15
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define protected amdgpu_kernel void @_Z12uavReadSpeedI15HIP_vector_typeIdLj2EEEvPT_S3_PKjjS2_(ptr addrspace(1) noalias nofree noundef readonly captures(none) %0, ptr addrspace(1) noalias nofree noundef writeonly captures(none) %1, ptr addrspace(1) noalias nofree noundef readonly captures(none) %2, i32 noundef %3, ptr addrspace(4) nofree noundef readnone byref(%struct.HIP_vector_type.0) align 16 captures(none) %4) local_unnamed_addr #0 comdat {
  %6 = tail call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !range !16, !invariant.load !17, !noundef !17
  %9 = zext nneg i16 %8 to i32
  %10 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %11 = mul i32 %10, %9
  %12 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %13 = add i32 %11, %12
  %14 = icmp eq i32 %3, 1
  br i1 %14, label %15, label %22

15:                                               ; preds = %5
  %16 = load i32, ptr addrspace(1) %2, align 4, !tbaa !18
  %17 = urem i32 %13, %16
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %18
  %20 = sext i32 %13 to i64
  %21 = getelementptr inbounds [16 x i8], ptr addrspace(1) %1, i64 %20
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 16 dereferenceable(16) %21, ptr addrspace(1) noundef align 16 dereferenceable(16) %19, i64 16, i1 false)
  br label %83

22:                                               ; preds = %5
  %23 = lshr i32 %3, 2
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %44, label %25

25:                                               ; preds = %22
  %26 = load i32, ptr addrspace(1) %2, align 4, !tbaa !18
  %27 = urem i32 %13, %26
  %28 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 12
  %29 = load <2 x i32>, ptr addrspace(1) %28, align 4, !tbaa !18
  %30 = insertelement <2 x i32> poison, i32 %27, i64 0
  %31 = shufflevector <2 x i32> %30, <2 x i32> poison, <2 x i32> zeroinitializer
  %32 = add <2 x i32> %29, %31
  %33 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 4
  %34 = load <2 x i32>, ptr addrspace(1) %33, align 4, !tbaa !18
  %35 = add <2 x i32> %34, %31
  %36 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 20
  %37 = load i32, ptr addrspace(1) %36, align 4, !tbaa !18
  %38 = insertelement <2 x i32> poison, i32 %37, i64 0
  %39 = shufflevector <2 x i32> %38, <2 x i32> poison, <2 x i32> zeroinitializer
  br label %51

40:                                               ; preds = %51
  %41 = fadd contract <2 x double> %63, %68
  %42 = fadd contract <2 x double> %73, %41
  %43 = fadd contract <2 x double> %78, %42
  br label %44

44:                                               ; preds = %40, %22
  %45 = phi <2 x double> [ zeroinitializer, %22 ], [ %43, %40 ]
  %46 = extractelement <2 x double> %45, i64 0
  %47 = extractelement <2 x double> %45, i64 1
  %48 = sext i32 %13 to i64
  %49 = getelementptr inbounds [16 x i8], ptr addrspace(1) %1, i64 %48
  store double %46, ptr addrspace(1) %49, align 16
  %50 = getelementptr inbounds nuw i8, ptr addrspace(1) %49, i64 8
  store double %47, ptr addrspace(1) %50, align 8
  br label %83

51:                                               ; preds = %25, %51
  %52 = phi i32 [ 0, %25 ], [ %81, %51 ]
  %53 = phi <2 x double> [ zeroinitializer, %25 ], [ %78, %51 ]
  %54 = phi <2 x double> [ zeroinitializer, %25 ], [ %73, %51 ]
  %55 = phi <2 x double> [ zeroinitializer, %25 ], [ %68, %51 ]
  %56 = phi <2 x double> [ zeroinitializer, %25 ], [ %63, %51 ]
  %57 = phi <2 x i32> [ %35, %25 ], [ %79, %51 ]
  %58 = phi <2 x i32> [ %32, %25 ], [ %80, %51 ]
  %59 = extractelement <2 x i32> %57, i64 0
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %60
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16, !tbaa !19
  %63 = fadd contract <2 x double> %56, %62
  %64 = extractelement <2 x i32> %57, i64 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %65
  %67 = load <2 x double>, ptr addrspace(1) %66, align 16, !tbaa !19
  %68 = fadd contract <2 x double> %55, %67
  %69 = extractelement <2 x i32> %58, i64 0
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %70
  %72 = load <2 x double>, ptr addrspace(1) %71, align 16, !tbaa !19
  %73 = fadd contract <2 x double> %54, %72
  %74 = extractelement <2 x i32> %58, i64 1
  %75 = zext i32 %74 to i64
  %76 = getelementptr inbounds nuw [16 x i8], ptr addrspace(1) %0, i64 %75
  %77 = load <2 x double>, ptr addrspace(1) %76, align 16, !tbaa !19
  %78 = fadd contract <2 x double> %53, %77
  %79 = add <2 x i32> %39, %57
  %80 = add <2 x i32> %39, %58
  %81 = add nuw nsw i32 %52, 1
  %82 = icmp eq i32 %81, %23
  br i1 %82, label %40, label %51, !llvm.loop !22

83:                                               ; preds = %44, %15
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!22 = distinct !{!22, !21}
