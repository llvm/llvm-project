; RUN: opt -mtriple=x86_64-pc-linux-gnu -pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s

; Make sure calls to the objc intrinsics are translated to calls in to the
; runtime

declare ptr @foo()
declare i32 @__gxx_personality_v0(...)

define ptr @test_objc_autorelease(ptr %arg0) {
; CHECK-LABEL: test_objc_autorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = notail call ptr @objc_autorelease(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.autorelease(ptr %arg0)
	ret ptr %0
}

define void @test_objc_autoreleasePoolPop(ptr %arg0) {
; CHECK-LABEL: test_objc_autoreleasePoolPop
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_autoreleasePoolPop(ptr %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.autoreleasePoolPop(ptr %arg0)
  ret void
}

define ptr @test_objc_autoreleasePoolPush() {
; CHECK-LABEL: test_objc_autoreleasePoolPush
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_autoreleasePoolPush()
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.autoreleasePoolPush()
	ret ptr %0
}

define ptr @test_objc_autoreleaseReturnValue(ptr %arg0) {
; CHECK-LABEL: test_objc_autoreleaseReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call ptr @objc_autoreleaseReturnValue(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.autoreleaseReturnValue(ptr %arg0)
	ret ptr %0
}

define void @test_objc_copyWeak(ptr %arg0, ptr %arg1) {
; CHECK-LABEL: test_objc_copyWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_copyWeak(ptr %arg0, ptr %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.copyWeak(ptr %arg0, ptr %arg1)
  ret void
}

define void @test_objc_destroyWeak(ptr %arg0) {
; CHECK-LABEL: test_objc_destroyWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_destroyWeak(ptr %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.destroyWeak(ptr %arg0)
  ret void
}

define ptr @test_objc_initWeak(ptr %arg0, ptr %arg1) {
; CHECK-LABEL: test_objc_initWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_initWeak(ptr %arg0, ptr %arg1)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.initWeak(ptr %arg0, ptr %arg1)
	ret ptr %0
}

define ptr @test_objc_loadWeak(ptr %arg0) {
; CHECK-LABEL: test_objc_loadWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_loadWeak(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.loadWeak(ptr %arg0)
	ret ptr %0
}

define ptr @test_objc_loadWeakRetained(ptr %arg0) {
; CHECK-LABEL: test_objc_loadWeakRetained
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_loadWeakRetained(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.loadWeakRetained(ptr %arg0)
	ret ptr %0
}

define void @test_objc_moveWeak(ptr %arg0, ptr %arg1) {
; CHECK-LABEL: test_objc_moveWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_moveWeak(ptr %arg0, ptr %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.moveWeak(ptr %arg0, ptr %arg1)
  ret void
}

define void @test_objc_release(ptr %arg0) {
; CHECK-LABEL: test_objc_release
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_release(ptr %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.release(ptr %arg0)
  ret void
}

define ptr @test_objc_retain(ptr %arg0) {
; CHECK-LABEL: test_objc_retain
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call ptr @objc_retain(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retain(ptr %arg0)
	ret ptr %0
}

define ptr @test_objc_retainAutorelease(ptr %arg0) {
; CHECK-LABEL: test_objc_retainAutorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_retainAutorelease(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retainAutorelease(ptr %arg0)
	ret ptr %0
}

define ptr @test_objc_retainAutoreleaseReturnValue(ptr %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleaseReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call ptr @objc_retainAutoreleaseReturnValue(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = tail call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr %arg0)
	ret ptr %0
}

define ptr @test_objc_retainAutoreleasedReturnValue(ptr %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleasedReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call ptr @objc_retainAutoreleasedReturnValue(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %arg0)
	ret ptr %0
}

define void @test_objc_retainAutoreleasedReturnValue_bundle() {
; CHECK-LABEL: test_objc_retainAutoreleasedReturnValue_bundle(
; CHECK-NEXT: call ptr @foo() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

define void @test_objc_retainAutoreleasedReturnValue_bundle_invoke() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: test_objc_retainAutoreleasedReturnValue_bundle_invoke(
; CHECK-NEXT: entry
; CHECK-NEXT: invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
entry:
  invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
      to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %1
}

define ptr @test_objc_retainBlock(ptr %arg0) {
; CHECK-LABEL: test_objc_retainBlock
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_retainBlock(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retainBlock(ptr %arg0)
	ret ptr %0
}

define void @test_objc_storeStrong(ptr %arg0, ptr %arg1) {
; CHECK-LABEL: test_objc_storeStrong
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_storeStrong(ptr %arg0, ptr %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.storeStrong(ptr %arg0, ptr %arg1)
	ret void
}

define ptr @test_objc_storeWeak(ptr %arg0, ptr %arg1) {
; CHECK-LABEL: test_objc_storeWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_storeWeak(ptr %arg0, ptr %arg1)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.storeWeak(ptr %arg0, ptr %arg1)
	ret ptr %0
}

define ptr @test_objc_unsafeClaimAutoreleasedReturnValue(ptr %arg0) {
; CHECK-LABEL: test_objc_unsafeClaimAutoreleasedReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call ptr @objc_unsafeClaimAutoreleasedReturnValue(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %arg0)
  ret ptr %0
}

define void @test_objc_unsafeClaimAutoreleasedReturnValue_bundle() {
; CHECK-LABEL: test_objc_unsafeClaimAutoreleasedReturnValue_bundle(
; CHECK-NEXT: call ptr @foo() [ "clang.arc.attachedcall"(ptr @objc_unsafeClaimAutoreleasedReturnValue) ]
  call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

define ptr @test_objc_retainedObject(ptr %arg0) {
; CHECK-LABEL: test_objc_retainedObject
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_retainedObject(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retainedObject(ptr %arg0)
  ret ptr %0
}

define ptr @test_objc_unretainedObject(ptr %arg0) {
; CHECK-LABEL: test_objc_unretainedObject
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_unretainedObject(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.unretainedObject(ptr %arg0)
  ret ptr %0
}

define ptr @test_objc_unretainedPointer(ptr %arg0) {
; CHECK-LABEL: test_objc_unretainedPointer
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_unretainedPointer(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.unretainedPointer(ptr %arg0)
  ret ptr %0
}

define ptr @test_objc_retain_autorelease(ptr %arg0) {
; CHECK-LABEL: test_objc_retain_autorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call ptr @objc_retain_autorelease(ptr %arg0)
; CHECK-NEXT: ret ptr %0
entry:
  %0 = call ptr @llvm.objc.retain.autorelease(ptr %arg0)
  ret ptr %0
}

define i32 @test_objc_sync_enter(ptr %arg0) {
; CHECK-LABEL: test_objc_sync_enter
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i32 @objc_sync_enter(ptr %arg0)
; CHECK-NEXT: ret i32 %0
entry:
  %0 = call i32 @llvm.objc.sync.enter(ptr %arg0)
  ret i32 %0
}

define i32 @test_objc_sync_exit(ptr %arg0) {
; CHECK-LABEL: test_objc_sync_exit
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i32 @objc_sync_exit(ptr %arg0)
; CHECK-NEXT: ret i32 %0
entry:
  %0 = call i32 @llvm.objc.sync.exit(ptr %arg0)
  ret i32 %0
}

declare ptr @llvm.objc.autorelease(ptr)
declare void @llvm.objc.autoreleasePoolPop(ptr)
declare ptr @llvm.objc.autoreleasePoolPush()
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare void @llvm.objc.copyWeak(ptr, ptr)
declare void @llvm.objc.destroyWeak(ptr)
declare extern_weak ptr @llvm.objc.initWeak(ptr, ptr)
declare ptr @llvm.objc.loadWeak(ptr)
declare ptr @llvm.objc.loadWeakRetained(ptr)
declare void @llvm.objc.moveWeak(ptr, ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.retainAutorelease(ptr)
declare ptr @llvm.objc.retainAutoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.retainBlock(ptr)
declare void @llvm.objc.storeStrong(ptr, ptr)
declare ptr @llvm.objc.storeWeak(ptr, ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.retainedObject(ptr)
declare ptr @llvm.objc.unretainedObject(ptr)
declare ptr @llvm.objc.unretainedPointer(ptr)
declare ptr @llvm.objc.retain.autorelease(ptr)
declare i32 @llvm.objc.sync.enter(ptr)
declare i32 @llvm.objc.sync.exit(ptr)

attributes #0 = { nounwind }

; CHECK: declare ptr @objc_autorelease(ptr)
; CHECK: declare void @objc_autoreleasePoolPop(ptr)
; CHECK: declare ptr @objc_autoreleasePoolPush()
; CHECK: declare ptr @objc_autoreleaseReturnValue(ptr)
; CHECK: declare void @objc_copyWeak(ptr, ptr)
; CHECK: declare void @objc_destroyWeak(ptr)
; CHECK: declare extern_weak ptr @objc_initWeak(ptr, ptr)
; CHECK: declare ptr @objc_loadWeak(ptr)
; CHECK: declare ptr @objc_loadWeakRetained(ptr)
; CHECK: declare void @objc_moveWeak(ptr, ptr)
; CHECK: declare void @objc_release(ptr) [[NLB:#[0-9]+]]
; CHECK: declare ptr @objc_retain(ptr) [[NLB]]
; CHECK: declare ptr @objc_retainAutorelease(ptr)
; CHECK: declare ptr @objc_retainAutoreleaseReturnValue(ptr)
; CHECK: declare ptr @objc_retainAutoreleasedReturnValue(ptr)
; CHECK: declare ptr @objc_retainBlock(ptr)
; CHECK: declare void @objc_storeStrong(ptr, ptr)
; CHECK: declare ptr @objc_storeWeak(ptr, ptr)
; CHECK: declare ptr @objc_unsafeClaimAutoreleasedReturnValue(ptr)
; CHECK: declare ptr @objc_retainedObject(ptr)
; CHECK: declare ptr @objc_unretainedObject(ptr)
; CHECK: declare ptr @objc_unretainedPointer(ptr)
; CHECK: declare ptr @objc_retain_autorelease(ptr)
; CHECK: declare i32 @objc_sync_enter(ptr)
; CHECK: declare i32 @objc_sync_exit(ptr)

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes [[NLB]] = { nonlazybind }
