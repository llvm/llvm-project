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
