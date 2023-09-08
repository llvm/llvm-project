; RUN: opt -S -passes=objc-arc < %s | FileCheck %s

declare void @use_pointer(ptr)
declare ptr @returner()
declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)

; Clean up residue left behind after inlining.

; CHECK-LABEL: define void @test0(
; CHECK: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(ptr %call.i) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %call.i) nounwind
  %1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %0) nounwind
  ret void
}

; Same as test0, but with slightly different use arrangements.

; CHECK-LABEL: define void @test1(
; CHECK: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test1(ptr %call.i) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %call.i) nounwind
  %1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call.i) nounwind
  ret void
}

; Delete a retainRV+autoreleaseRV even if the pointer is used.

; CHECK-LABEL: define void @test24(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @use_pointer(ptr %p)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test24(ptr %p) {
entry:
  call ptr @llvm.objc.autoreleaseReturnValue(ptr %p) nounwind
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %p) nounwind
  call void @use_pointer(ptr %p)
  ret void
}

; Check that we can delete the autoreleaseRV+retainAutoreleasedRV pair even in
; presence of instructions added by the inliner as part of the return sequence.

; 1) Noop instructions: bitcasts and zero-indices GEPs.

; CHECK-LABEL: define ptr @testNoop(
; CHECK: entry:
; CHECK-NEXT: ret ptr %call.i
; CHECK-NEXT: }
define ptr @testNoop(ptr %call.i) {
entry:
  %0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call.i) nounwind
  %1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call.i) nounwind
  ret ptr %call.i
}

; 2) Lifetime markers.

declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)

; CHECK-LABEL: define ptr @testLifetime(
; CHECK: entry:
; CHECK-NEXT: %obj = alloca i8
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr %obj)
; CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr %obj)
; CHECK-NEXT: ret ptr %call.i
; CHECK-NEXT: }
define ptr @testLifetime(ptr %call.i) {
entry:
  %obj = alloca i8
  call void @llvm.lifetime.start.p0(i64 8, ptr %obj)
  %0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call.i) nounwind
  call void @llvm.lifetime.end.p0(i64 8, ptr %obj)
  %1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call.i) nounwind
  ret ptr %call.i
}

; 3) Dynamic alloca markers.

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)

; CHECK-LABEL: define ptr @testStack(
; CHECK: entry:
; CHECK-NEXT: %save = tail call ptr @llvm.stacksave.p0()
; CHECK-NEXT: %obj = alloca i8, i8 %arg
; CHECK-NEXT: call void @llvm.stackrestore.p0(ptr %save)
; CHECK-NEXT: ret ptr %call.i
; CHECK-NEXT: }
define ptr @testStack(ptr %call.i, i8 %arg) {
entry:
  %save = tail call ptr @llvm.stacksave()
  %obj = alloca i8, i8 %arg
  %0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call.i) nounwind
  call void @llvm.stackrestore(ptr %save)
  %1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call.i) nounwind
  ret ptr %call.i
}
