; RUN: opt -S -passes='amdgpu-codegenprepare' -mtriple=amdgcn-amd-amdhsa -disable-verify < %s | FileCheck %s --check-prefix=DEREF
; RUN: opt -S -passes='loop-mssa(licm)' -mtriple=amdgcn-amd-amdhsa -disable-verify < %s | FileCheck %s --check-prefix=WITHOUT
; RUN: opt -S -passes='amdgpu-codegenprepare,loop-mssa(licm)' -mtriple=amdgcn-amd-amdhsa < %s | FileCheck %s --check-prefix=WITH

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define protected amdgpu_kernel void @foo(ptr addrspace(1) noundef readonly captures(none) %d_a.coerce, ptr addrspace(1) noundef readonly captures(none) %d_b.coerce, ptr addrspace(1) noundef writeonly captures(none) %d_c.coerce, i32 noundef %count) local_unnamed_addr #0 {
; DEREF-LABEL: define protected amdgpu_kernel void @foo(
; DEREF:    tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
;
; WITHOUT-LABEL: define protected amdgpu_kernel void @foo(
; WITHOUT-SAME: ptr addrspace(1) noundef readonly captures(none) [[D_A_COERCE:%.*]], ptr addrspace(1) noundef readonly captures(none) [[D_B_COERCE:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[D_C_COERCE:%.*]], i32 noundef [[COUNT:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
; WITHOUT-NEXT:  [[ENTRY:.*:]]
; WITHOUT-NEXT:    [[TMP0:%.*]] = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y()
; WITHOUT-NEXT:    [[CMP11:%.*]] = icmp samesign ult i32 [[TMP0]], 4
; WITHOUT-NEXT:    br i1 [[CMP11]], label %[[FOR_BODY_LR_PH:.*]], label %[[FOR_COND_CLEANUP:.*]]
; WITHOUT:       [[FOR_BODY_LR_PH]]:
; WITHOUT-NEXT:    [[TMP1:%.*]] = tail call i32 @llvm.amdgcn.workgroup.id.x()
; WITHOUT-NEXT:    [[TMP2:%.*]] = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
; WITHOUT-NEXT:    [[CMP79:%.*]] = icmp samesign ult i32 [[TMP2]], 4
; WITHOUT-NEXT:    [[CONV:%.*]] = sext i32 [[TMP1]] to i64
; WITHOUT-NEXT:    [[MUL:%.*]] = mul nsw i64 [[CONV]], 192
; WITHOUT-NEXT:    [[TMP3:%.*]] = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; WITHOUT-NEXT:    [[TMP4:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(4) [[TMP3]], i64 12
; WITHOUT-NEXT:    [[TMP5:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(4) [[TMP3]], i64 14
; WITHOUT-NEXT:    [[DOTIN_I_I_I:%.*]] = load i16, ptr addrspace(4) [[TMP5]], align 2, !tbaa [[SHORT_TBAA6:![0-9]+]]
; WITHOUT-NEXT:    [[CONV_I_I:%.*]] = zext i16 [[DOTIN_I_I_I]] to i32
; WITHOUT-NEXT:    br label %[[FOR_BODY:.*]]
;
; WITH-LABEL: define protected amdgpu_kernel void @foo(
; WITH-SAME: ptr addrspace(1) noundef readonly captures(none) [[D_A_COERCE:%.*]], ptr addrspace(1) noundef readonly captures(none) [[D_B_COERCE:%.*]], ptr addrspace(1) noundef writeonly captures(none) [[D_C_COERCE:%.*]], i32 noundef [[COUNT:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
; WITH-NEXT:  [[ENTRY:.*:]]
; WITH-NEXT:    [[TMP0:%.*]] = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y()
; WITH-NEXT:    [[CMP11:%.*]] = icmp samesign ult i32 [[TMP0]], 4
; WITH-NEXT:    br i1 [[CMP11]], label %[[FOR_BODY_LR_PH:.*]], label %[[FOR_COND_CLEANUP:.*]]
; WITH:       [[FOR_BODY_LR_PH]]:
; WITH-NEXT:    [[TMP1:%.*]] = tail call i32 @llvm.amdgcn.workgroup.id.x()
; WITH-NEXT:    [[TMP2:%.*]] = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
; WITH-NEXT:    [[CMP79:%.*]] = icmp samesign ult i32 [[TMP2]], 4
; WITH-NEXT:    [[CONV:%.*]] = sext i32 [[TMP1]] to i64
; WITH-NEXT:    [[MUL:%.*]] = mul nsw i64 [[CONV]], 192
; WITH-NEXT:    [[TMP3:%.*]] = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; WITH-NEXT:    [[TMP4:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(4) [[TMP3]], i64 12
; WITH-NEXT:    [[TMP5:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(4) [[TMP3]], i64 14
; WITH-NEXT:    [[DOTIN_I_I_I:%.*]] = load i16, ptr addrspace(4) [[TMP5]], align 2, !tbaa [[SHORT_TBAA6:![0-9]+]]
; WITH-NEXT:    [[CONV_I_I:%.*]] = zext i16 [[DOTIN_I_I_I]] to i32
; WITH-NEXT:    [[DOTIN_I_I_I7:%.*]] = load i16, ptr addrspace(4) [[TMP4]], align 4
; WITH-NEXT:    [[CONV_I_I8:%.*]] = zext i16 [[DOTIN_I_I_I7]] to i32
; WITH-NEXT:    br label %[[FOR_BODY:.*]]
;
entry:
  %0 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y()
  %cmp11 = icmp samesign ult i32 %0, 4
  br i1 %cmp11, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %1 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %2 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %cmp79 = icmp samesign ult i32 %2, 4
  %conv = sext i32 %1 to i64
  %mul = mul nsw i64 %conv, 192
  %3 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %4 = getelementptr inbounds nuw i8, ptr addrspace(4) %3, i64 12
  %5 = getelementptr inbounds nuw i8, ptr addrspace(4) %3, i64 14
  %.in.i.i.i = load i16, ptr addrspace(4) %5, align 2, !tbaa !12
  %conv.i.i = zext i16 %.in.i.i.i to i32
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup8, %entry
  ret void

for.body:                                         ; preds = %for.cond.cleanup8, %for.body.lr.ph
  %thread_y.012 = phi i32 [ %0, %for.body.lr.ph ], [ %add21, %for.cond.cleanup8 ]
  br i1 %cmp79, label %for.body9.lr.ph, label %for.cond.cleanup8

for.body9.lr.ph:                                  ; preds = %for.body
  %mul10 = shl nuw nsw i32 %thread_y.012, 2
  %conv11 = zext nneg i32 %mul10 to i64
  %add = add nuw nsw i64 %mul, %conv11
  %.in.i.i.i7 = load i16, ptr addrspace(4) %4, align 4, !tbaa !12
  %conv.i.i8 = zext i16 %.in.i.i.i7 to i32
  br label %for.body9

for.cond.cleanup8:                                ; preds = %for.body9, %for.body
  %add21 = add nuw nsw i32 %thread_y.012, %conv.i.i
  %cmp = icmp samesign ult i32 %add21, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !16

for.body9:                                        ; preds = %for.body9, %for.body9.lr.ph
  %thread_x.010 = phi i32 [ %2, %for.body9.lr.ph ], [ %add18, %for.body9 ]
  %conv12 = zext nneg i32 %thread_x.010 to i64
  %add13 = add nuw nsw i64 %add, %conv12
  %arrayidx = getelementptr inbounds double, ptr addrspace(1) %d_a.coerce, i64 %add13
  %6 = load double, ptr addrspace(1) %arrayidx, align 8, !tbaa !18
  %arrayidx14 = getelementptr inbounds double, ptr addrspace(1) %d_b.coerce, i64 %add13
  %7 = load double, ptr addrspace(1) %arrayidx14, align 8, !tbaa !18
  %add15 = fadd contract double %6, %7
  %arrayidx16 = getelementptr inbounds double, ptr addrspace(1) %d_c.coerce, i64 %add13
  store double %add15, ptr addrspace(1) %arrayidx16, align 8, !tbaa !18
  %add18 = add nuw nsw i32 %thread_x.010, %conv.i.i8
  %cmp7 = icmp samesign ult i32 %add18, 4
  br i1 %cmp7, label %for.body9, label %for.cond.cleanup8, !llvm.loop !20
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #1


; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,192" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.errno.tbaa = !{!7}
!opencl.ocl.version = !{!11}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{i32 2, i32 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
!18 = !{!19, !19, i64 0}
!19 = !{!"double", !9, i64 0}
!20 = distinct !{!20, !17}
