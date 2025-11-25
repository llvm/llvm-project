; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=instcombine %s | FileCheck %s

@test.data = private unnamed_addr addrspace(2) constant [8 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7], align 4
@test.ptrdata = private unnamed_addr addrspace(2) constant [8 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null], align 8

; Verify that InstCombine copies range metadata when cloning a load as part of
; replacing an alloca initialized via memcpy from a constant in another
; address space. OK
define i32 @copy_range_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_range_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !range [[RNG0:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !range !0
  ret i32 %l
}



declare void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1)

!0 = !{i32 0, i32 100}

; Verify TBAA metadata on a cloned load is preserved. OK
define i32 @copy_tbaa_metadata_after_memcpy(i64 %x, ptr addrspace(5) %sink) {
; CHECK-LABEL: define i32 @copy_tbaa_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]], ptr addrspace(5) [[SINK:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !tbaa [[TBAA0:![0-9]+]]
; CHECK-NEXT:    store i32 [[L]], ptr addrspace(5) [[SINK]], align 4
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !tbaa !1
  store i32 %l, ptr addrspace(5) %sink, align 4
  ret i32 %l
}

!1 = !{!2, !2, i64 0}
!2 = !{!"scalar type", !3}
!3 = !{!"root"}

; CHECK: [[TBAA0]] = !{[[TBAATY:![0-9]+]], [[TBAATY]], i64 0}
; CHECK: [[TBAATY]] = !{!"scalar type", [[TBAAROOT:![0-9]+]]}
; CHECK: [[TBAAROOT]] = !{!"root"}

; Verify dereferenceable_or_null metadata on a cloned load is preserved
; when the loaded value type is a pointer. OK
define ptr @copy_deref_or_null_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define ptr @copy_deref_or_null_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds ptr, ptr addrspace(2) @test.ptrdata, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load ptr, ptr addrspace(2) [[ARRAYIDX]], align 8, !dereferenceable_or_null [[DEREF_OR_NULL:![0-9]+]]
; CHECK-NEXT:    ret ptr [[L]]
;
entry:
  %data = alloca [8 x ptr], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 8 %data, ptr addrspace(2) align 8 @test.ptrdata, i64 64, i1 false)
  %arrayidx = getelementptr inbounds [8 x ptr], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load ptr, ptr addrspace(5) %arrayidx, align 8, !dereferenceable_or_null !4
  ret ptr %l
}

!4 = !{i64 8}

; CHECK: [[DEREF_OR_NULL]] = !{i64 8}

; Verify nonnull metadata on a cloned load is preserved
; when the loaded value type is a pointer. OK
define ptr @copy_nonnull_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define ptr @copy_nonnull_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds ptr, ptr addrspace(2) @test.ptrdata, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load ptr, ptr addrspace(2) [[ARRAYIDX]], align 8, !nonnull [[NONNULL:![0-9]+]]
; CHECK-NEXT:    ret ptr [[L]]
;
entry:
  %data = alloca [8 x ptr], align 8, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 8 %data, ptr addrspace(2) align 8 @test.ptrdata, i64 64, i1 false)
  %arrayidx = getelementptr inbounds [8 x ptr], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load ptr, ptr addrspace(5) %arrayidx, align 8, !nonnull !5
  ret ptr %l
}

!5 = !{}

; CHECK: [[NONNULL]] = !{}

; Verify invariant.load metadata on a cloned load is preserved. OK
define i32 @copy_invariant_load_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_invariant_load_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !invariant.load [[INVLOAD:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !invariant.load !5
  ret i32 %l
}

; CHECK: [[INVLOAD]] = !{}

; Verify alias.scope and noalias metadata on a cloned load are preserved. OK
define i32 @copy_aliasscope_noalias_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_aliasscope_noalias_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !alias.scope [[ALIASSET:![0-9]+]], !noalias [[ALIASSET]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !alias.scope !6, !noalias !6
  ret i32 %l
}

; Verify nontemporal metadata on a cloned load is preserved.OK
define i32 @copy_nontemporal_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_nontemporal_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !nontemporal [[NT:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !nontemporal !9
  ret i32 %l
}

; Verify access group metadata on a cloned load is preserved. OK
define i32 @copy_access_group_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_access_group_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !llvm.access.group [[ACCGRP:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !llvm.access.group !10
  ret i32 %l
}

; Verify noalias.addrspace metadata on a cloned load is preserved.
define i32 @copy_noalias_addrspace_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_noalias_addrspace_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !noalias.addrspace [[NAAS:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !noalias.addrspace !12
  ret i32 %l
}

; Verify llvm.mem.parallel_loop_access metadata on a cloned load is preserved. OK
define i32 @copy_mem_parallel_loop_access_metadata_after_memcpy(i64 %x) {
; CHECK-LABEL: define i32 @copy_mem_parallel_loop_access_metadata_after_memcpy(
; CHECK-SAME: i64 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr addrspace(2) @test.data, i64 [[X]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr addrspace(2) [[ARRAYIDX]], align 4, !llvm.mem.parallel_loop_access [[MPLA:![0-9]+]]
; CHECK-NEXT:    ret i32 [[L]]
;
entry:
  %data = alloca [8 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p2.i64(ptr addrspace(5) align 4 %data, ptr addrspace(2) align 4 @test.data, i64 32, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], ptr addrspace(5) %data, i64 0, i64 %x
  %l = load i32, ptr addrspace(5) %arrayidx, align 4, !llvm.mem.parallel_loop_access !13
  ret i32 %l
}

!6 = !{!7}
!7 = distinct !{!7, !8}
!8 = distinct !{!8}
!9 = !{i32 1}
!10 = distinct !{}
!12 = !{i32 5, i32 6}
!13 = !{!14}
!14 = distinct !{}

; CHECK: [[ALIASSET]] = !{[[ALIASSETNODE:![0-9]+]]}
; CHECK: [[ALIASSETNODE]] = distinct !{[[ALIASSETNODE]], [[ALIASSETNODE2:![0-9]+]]}
; CHECK: [[ALIASSETNODE2]] = distinct !{[[ALIASSETNODE2]]}
; CHECK: [[NT]] = !{i32 1}
; CHECK: [[ACCGRP]] = distinct !{}
; CHECK: [[NOUNDEF]] = !{}
; CHECK: [[NAAS]] = !{i32 5, i32 6}
; CHECK: [[MPLA]] = !{[[MPLALOOP:![0-9]+]]}
; CHECK: [[MPLALOOP]] = distinct !{}
