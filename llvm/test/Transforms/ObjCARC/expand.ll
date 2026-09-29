; RUN: opt -passes=objc-arc-expand -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.autorelease(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutorelease(ptr)
declare ptr @llvm.objc.retainAutoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainBlock(ptr)

declare void @use_pointer(ptr)

; CHECK: define void @test_retain(ptr %x) [[NUW:#[0-9]+]] {
; CHECK: call ptr @llvm.objc.retain(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_retain(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

; CHECK: define void @test_retainAutoreleasedReturnValue(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_retainAutoreleasedReturnValue(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

; CHECK: define void @test_retainAutorelease(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.retainAutorelease(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_retainAutorelease(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retainAutorelease(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

; CHECK: define void @test_retainAutoreleaseReturnValue(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_retainAutoreleaseReturnValue(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

; CHECK: define void @test_autorelease(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.autorelease(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_autorelease(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.autorelease(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

; CHECK: define void @test_autoreleaseReturnValue(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
; CHECK: call void @use_pointer(ptr %x)
; CHECK: }
define void @test_autoreleaseReturnValue(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; RetainBlock is not strictly forwarding. Do not touch it. ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK: define void @test_retainBlock(ptr %x) [[NUW]] {
; CHECK: call ptr @llvm.objc.retainBlock(ptr %x)
; CHECK: call void @use_pointer(ptr %0)
; CHECK: }
define void @test_retainBlock(ptr %x) nounwind {
entry:
  %0 = call ptr @llvm.objc.retainBlock(ptr %x) nounwind
  call void @use_pointer(ptr %0)
  ret void
}
