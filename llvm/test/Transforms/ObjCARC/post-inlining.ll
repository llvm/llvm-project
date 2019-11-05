; RUN: opt -S -objc-arc < %s | FileCheck %s

declare void @use_pointer(i8*)
declare i8* @returner()
declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)

; Clean up residue left behind after inlining.

; CHECK-LABEL: define void @test0(
; CHECK: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(i8* %call.i) {
entry:
  %0 = tail call i8* @llvm.objc.retain(i8* %call.i) nounwind
  %1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %0) nounwind
  ret void
}

; Same as test0, but with slightly different use arrangements.

; CHECK-LABEL: define void @test1(
; CHECK: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test1(i8* %call.i) {
entry:
  %0 = tail call i8* @llvm.objc.retain(i8* %call.i) nounwind
  %1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call.i) nounwind
  ret void
}

; Delete a retainRV+autoreleaseRV even if the pointer is used.

; CHECK-LABEL: define void @test24(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @use_pointer(i8* %p)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test24(i8* %p) {
entry:
  call i8* @llvm.objc.autoreleaseReturnValue(i8* %p) nounwind
  call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %p) nounwind
  call void @use_pointer(i8* %p)
  ret void
}

; Check that we can delete the autoreleaseRV+retainAutoreleasedRV pair even in
; presence of instructions added by the inliner as part of the return sequence.

; 1) Noop instructions: bitcasts and zero-indices GEPs.

; CHECK-LABEL: define i8* @testNoop(
; CHECK: entry:
; CHECK-NEXT: %noop0 = bitcast i8* %call.i to i64*
; CHECK-NEXT: %noop1 = getelementptr i8, i8* %call.i, i32 0
; CHECK-NEXT: ret i8* %call.i
; CHECK-NEXT: }
define i8* @testNoop(i8* %call.i) {
entry:
  %0 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call.i) nounwind
  %noop0 = bitcast i8* %call.i to i64*
  %noop1 = getelementptr i8, i8* %call.i, i32 0
  %1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call.i) nounwind
  ret i8* %call.i
}

; 2) Lifetime markers.

declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)

; CHECK-LABEL: define i8* @testLifetime(
; CHECK: entry:
; CHECK-NEXT: %obj = alloca i8
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* %obj)
; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* %obj)
; CHECK-NEXT: ret i8* %call.i
; CHECK-NEXT: }
define i8* @testLifetime(i8* %call.i) {
entry:
  %obj = alloca i8
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %obj)
  %0 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call.i) nounwind
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %obj)
  %1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call.i) nounwind
  ret i8* %call.i
}

; 3) Dynamic alloca markers.

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

; CHECK-LABEL: define i8* @testStack(
; CHECK: entry:
; CHECK-NEXT: %save = tail call i8* @llvm.stacksave()
; CHECK-NEXT: %obj = alloca i8, i8 %arg
; CHECK-NEXT: call void @llvm.stackrestore(i8* %save)
; CHECK-NEXT: ret i8* %call.i
; CHECK-NEXT: }
define i8* @testStack(i8* %call.i, i8 %arg) {
entry:
  %save = tail call i8* @llvm.stacksave()
  %obj = alloca i8, i8 %arg
  %0 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call.i) nounwind
  call void @llvm.stackrestore(i8* %save)
  %1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call.i) nounwind
  ret i8* %call.i
}
