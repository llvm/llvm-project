; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)

; Retain must not sink past an atomicrmw with release or stronger ordering.

define void @test_atomicrmw_release(ptr %obj, ptr %atomic_slot) {
; CHECK-LABEL: @test_atomicrmw_release
; CHECK: call ptr @llvm.objc.retain
; CHECK: atomicrmw
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %old_value = atomicrmw xchg ptr %atomic_slot, i64 %obj_as_int release, align 8
  %old_obj = inttoptr i64 %old_value to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

define void @test_atomicrmw_acqrel(ptr %obj, ptr %atomic_slot) {
; CHECK-LABEL: @test_atomicrmw_acqrel
; CHECK: call ptr @llvm.objc.retain
; CHECK: atomicrmw
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %old_value = atomicrmw xchg ptr %atomic_slot, i64 %obj_as_int acq_rel, align 8
  %old_obj = inttoptr i64 %old_value to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

define void @test_atomicrmw_seqcst(ptr %obj, ptr %atomic_slot) {
; CHECK-LABEL: @test_atomicrmw_seqcst
; CHECK: call ptr @llvm.objc.retain
; CHECK: atomicrmw
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %old_value = atomicrmw xchg ptr %atomic_slot, i64 %obj_as_int seq_cst, align 8
  %old_obj = inttoptr i64 %old_value to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

; Retain must not sink past a cmpxchg with release or stronger success ordering.

define void @test_cmpxchg_release(ptr %obj, ptr %atomic_slot, i64 %expected) {
; CHECK-LABEL: @test_cmpxchg_release
; CHECK: call ptr @llvm.objc.retain
; CHECK: cmpxchg
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %res = cmpxchg ptr %atomic_slot, i64 %expected, i64 %obj_as_int release seq_cst, align 8
  %old_int = extractvalue { i64, i1 } %res, 0
  %old_obj = inttoptr i64 %old_int to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

define void @test_cmpxchg_acqrel(ptr %obj, ptr %atomic_slot, i64 %expected) {
; CHECK-LABEL: @test_cmpxchg_acqrel
; CHECK: call ptr @llvm.objc.retain
; CHECK: cmpxchg
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %res = cmpxchg ptr %atomic_slot, i64 %expected, i64 %obj_as_int acq_rel seq_cst, align 8
  %old_int = extractvalue { i64, i1 } %res, 0
  %old_obj = inttoptr i64 %old_int to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

define void @test_cmpxchg_seqcst(ptr %obj, ptr %atomic_slot, i64 %expected) {
; CHECK-LABEL: @test_cmpxchg_seqcst
; CHECK: call ptr @llvm.objc.retain
; CHECK: cmpxchg
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %res = cmpxchg ptr %atomic_slot, i64 %expected, i64 %obj_as_int seq_cst seq_cst, align 8
  %old_int = extractvalue { i64, i1 } %res, 0
  %old_obj = inttoptr i64 %old_int to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}
