; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)

; Retain must not sink past atomicrmw.
define void @test_atomicrmw(ptr %obj, ptr %atomic_slot) {
; CHECK-LABEL: @test_atomicrmw
; CHECK: call ptr @llvm.objc.retain
; CHECK: atomicrmw
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %old_value = atomicrmw xchg ptr %atomic_slot, i64 %obj_as_int monotonic, align 8
  %old_obj = inttoptr i64 %old_value to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

; Retain must not sink past cmpxchg.
define void @test_cmpxchg(ptr %obj, ptr %atomic_slot, i64 %expected) {
; CHECK-LABEL: @test_cmpxchg
; CHECK: call ptr @llvm.objc.retain
; CHECK: cmpxchg
entry:
  %obj_as_int = ptrtoint ptr %obj to i64
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  %res = cmpxchg ptr %atomic_slot, i64 %expected, i64 %obj_as_int monotonic monotonic, align 8
  %old_int = extractvalue { i64, i1 } %res, 0
  %old_obj = inttoptr i64 %old_int to ptr
  call void @llvm.objc.release(ptr %old_obj)
  call void @llvm.objc.release(ptr %obj)
  ret void
}

; Retain must not sink past an atomic store.
define void @test_atomic_store(ptr %obj, ptr %slot) {
; CHECK-LABEL: @test_atomic_store
; CHECK: call ptr @llvm.objc.retain
; CHECK: store atomic
entry:
  %retained = call ptr @llvm.objc.retain(ptr %obj)
  store atomic ptr %obj, ptr %slot monotonic, align 8
  call void @llvm.objc.release(ptr %obj)
  ret void
}
