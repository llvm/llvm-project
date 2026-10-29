; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 %s -o - | FileCheck %s

; CodeGenPrepare sinks the flat-to-global addrspacecasts to the phi
; predecessors. Each uniform pointer is then copied into the VGPR used by a
; byte store. That move has to run under the exec mask of its own predecessor.
; Between s_or_saveexec and the matching s_xor, exec covers both predecessors,
; so the move would publish one buffer base on both paths
; (ROCm/rocm-systems#11305).

; CHECK-LABEL: _Z6maskeri7array_tS_S_S_S_:
; CHECK: s_or_saveexec_b64 [[SAV:s\[[0-9]+:[0-9]+\]]], [[SAV]]
; CHECK-NOT: v_mov_b64
; CHECK: s_xor_b64 exec, exec, [[SAV]]

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.array_t = type { ptr, [4 x i32], i32, i32, [4 x i32] }


; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define protected amdgpu_kernel void @_Z6maskeri7array_tS_S_S_S_(i32 noundef %0, ptr addrspace(4) nocapture noundef readnone byref(%struct.array_t) align 8 %1, ptr addrspace(4) nocapture noundef readonly byref(%struct.array_t) align 8 %2, ptr addrspace(4) nocapture noundef readonly byref(%struct.array_t) align 8 %3, ptr addrspace(4) nocapture noundef readonly byref(%struct.array_t) align 8 %4, ptr addrspace(4) nocapture noundef readonly byref(%struct.array_t) align 8 %5) local_unnamed_addr #0 {
  %7 = load ptr, ptr addrspace(4) %2, align 8, !amdgpu.noclobber !6
  %8 = addrspacecast ptr %7 to ptr addrspace(1)
  %9 = load ptr, ptr addrspace(4) %3, align 8, !amdgpu.noclobber !6
  %10 = addrspacecast ptr %9 to ptr addrspace(1)
  %11 = load ptr, ptr addrspace(4) %4, align 8, !amdgpu.noclobber !6
  %12 = addrspacecast ptr %11 to ptr addrspace(1)
  %13 = getelementptr inbounds nuw i8, ptr addrspace(4) %4, i64 36
  %14 = load i32, ptr addrspace(4) %13, align 4, !amdgpu.noclobber !6
  %15 = getelementptr inbounds nuw i8, ptr addrspace(4) %4, i64 40
  %16 = load i32, ptr addrspace(4) %15, align 8, !amdgpu.noclobber !6
  %17 = load ptr, ptr addrspace(4) %5, align 8, !amdgpu.noclobber !6
  %18 = addrspacecast ptr %17 to ptr addrspace(1)
  %19 = getelementptr inbounds nuw i8, ptr addrspace(4) %5, i64 12
  %20 = load i32, ptr addrspace(4) %19, align 4, !amdgpu.noclobber !6
  %21 = getelementptr inbounds nuw i8, ptr addrspace(4) %5, i64 32
  %22 = load i32, ptr addrspace(4) %21, align 8, !amdgpu.noclobber !6
  %23 = getelementptr inbounds nuw i8, ptr addrspace(4) %5, i64 36
  %24 = load i32, ptr addrspace(4) %23, align 4, !amdgpu.noclobber !6
  %25 = getelementptr inbounds nuw i8, ptr addrspace(4) %5, i64 40
  %26 = load i32, ptr addrspace(4) %25, align 8, !amdgpu.noclobber !6
  %27 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %28 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %29 = getelementptr inbounds nuw i8, ptr addrspace(4) %28, i64 12
  %30 = load i16, ptr addrspace(4) %29, align 4, !tbaa !7
  %31 = zext i16 %30 to i32
  %32 = mul i32 %27, %31
  %33 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %34 = add i32 %32, %33
  %35 = icmp slt i32 %34, %0
  br i1 %35, label %36, label %102

36:                                               ; preds = %6
  %37 = getelementptr inbounds nuw i8, ptr addrspace(4) %5, i64 16
  %38 = load i32, ptr addrspace(4) %37, align 8, !amdgpu.noclobber !6
  %39 = and i32 %34, 31
  %40 = lshr i32 %34, 5
  %41 = and i32 %40, 31
  %42 = icmp slt i32 %39, %20
  %43 = icmp slt i32 %41, %38
  %44 = select i1 %42, i1 %43, i1 false
  br i1 %44, label %45, label %102

45:                                               ; preds = %36
  %46 = sext i32 %34 to i64
  %47 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 %46
  %48 = mul nsw i32 %39, %14
  %49 = mul nsw i32 %41, %16
  %50 = add nsw i32 %49, %48
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 %51
  %53 = mul nsw i32 %39, %24
  %54 = mul nsw i32 %41, %26
  %55 = add i32 %54, %53
  %56 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 %46
  %57 = add nuw nsw i32 %39, 1
  %58 = load i8, ptr addrspace(1) %47, align 1, !tbaa !11
  store i8 %58, ptr addrspace(1) %52, align 1, !tbaa !11
  %59 = icmp eq i32 %39, 0
  br i1 %59, label %60, label %64

60:                                               ; preds = %45
  %61 = sext i32 %55 to i64
  %62 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %61
  store i8 1, ptr addrspace(1) %62, align 1, !tbaa !11
  %63 = load i8, ptr addrspace(1) %47, align 1, !tbaa !11
  br label %77

64:                                               ; preds = %45
  %65 = load i8, ptr addrspace(1) %56, align 1, !tbaa !11
  %66 = icmp eq i8 %65, 0
  br i1 %66, label %77, label %67

67:                                               ; preds = %64
  %68 = mul nsw i32 %57, %24
  %69 = add i32 %54, %68
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %70
  store i8 1, ptr addrspace(1) %71, align 1, !tbaa !11
  %72 = load i8, ptr addrspace(1) %47, align 1, !tbaa !11
  %73 = mul nsw i32 %57, %14
  %74 = add nsw i32 %73, %49
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 %75
  store i8 %72, ptr addrspace(1) %76, align 1, !tbaa !11
  br label %77

77:                                               ; preds = %64, %67, %60
  %78 = phi i8 [ %58, %64 ], [ %72, %67 ], [ %63, %60 ]
  %79 = add nuw nsw i32 %39, 1
  %80 = add nsw i32 %39, -1
  store i8 %78, ptr addrspace(1) %52, align 1, !tbaa !11
  %81 = icmp samesign ult i32 %79, %20
  br i1 %81, label %84, label %82

82:                                               ; preds = %77
  %83 = add i32 %55, %22
  br label %96

84:                                               ; preds = %77
  %85 = load i8, ptr addrspace(1) %56, align 1, !tbaa !11
  %86 = icmp eq i8 %85, 0
  br i1 %86, label %102, label %87

87:                                               ; preds = %84
  %88 = mul nsw i32 %80, %24
  %89 = add i32 %22, %54
  %90 = add i32 %89, %88
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %91
  store i8 1, ptr addrspace(1) %92, align 1, !tbaa !11
  %93 = load i8, ptr addrspace(1) %47, align 1, !tbaa !11
  %94 = mul nsw i32 %80, %14
  %95 = add nsw i32 %94, %49
  br label %96

96:                                               ; preds = %82, %87
  %97 = phi i32 [ %95, %87 ], [ %83, %82 ]
  %98 = phi ptr addrspace(1) [ %12, %87 ], [ %18, %82 ]
  %99 = phi i8 [ %93, %87 ], [ 1, %82 ]
  %100 = sext i32 %97 to i64
  %101 = getelementptr inbounds i8, ptr addrspace(1) %98, i64 %100
  store i8 %99, ptr addrspace(1) %101, align 1, !tbaa !11
  br label %102

102:                                              ; preds = %84, %96, %36, %6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"}
!6 = !{}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
