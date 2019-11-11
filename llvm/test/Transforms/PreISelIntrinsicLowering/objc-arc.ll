; RUN: opt -pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s
; RUN: opt -passes='pre-isel-intrinsic-lowering' -S -o - %s | FileCheck %s

; Make sure calls to the objc intrinsics are translated to calls in to the
; runtime

define i8* @test_objc_autorelease(i8* %arg0) {
; CHECK-LABEL: test_objc_autorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = notail call i8* @objc_autorelease(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.autorelease(i8* %arg0)
	ret i8* %0
}

define void @test_objc_autoreleasePoolPop(i8* %arg0) {
; CHECK-LABEL: test_objc_autoreleasePoolPop
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_autoreleasePoolPop(i8* %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.autoreleasePoolPop(i8* %arg0)
  ret void
}

define i8* @test_objc_autoreleasePoolPush() {
; CHECK-LABEL: test_objc_autoreleasePoolPush
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_autoreleasePoolPush()
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.autoreleasePoolPush()
	ret i8* %0
}

define i8* @test_objc_autoreleaseReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_autoreleaseReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call i8* @objc_autoreleaseReturnValue(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %arg0)
	ret i8* %0
}

define void @test_objc_copyWeak(i8** %arg0, i8** %arg1) {
; CHECK-LABEL: test_objc_copyWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_copyWeak(i8** %arg0, i8** %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.copyWeak(i8** %arg0, i8** %arg1)
  ret void
}

define void @test_objc_destroyWeak(i8** %arg0) {
; CHECK-LABEL: test_objc_destroyWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_destroyWeak(i8** %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.destroyWeak(i8** %arg0)
  ret void
}

define i8* @test_objc_initWeak(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_initWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_initWeak(i8** %arg0, i8* %arg1)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.initWeak(i8** %arg0, i8* %arg1)
	ret i8* %0
}

define i8* @test_objc_loadWeak(i8** %arg0) {
; CHECK-LABEL: test_objc_loadWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_loadWeak(i8** %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.loadWeak(i8** %arg0)
	ret i8* %0
}

define i8* @test_objc_loadWeakRetained(i8** %arg0) {
; CHECK-LABEL: test_objc_loadWeakRetained
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_loadWeakRetained(i8** %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.loadWeakRetained(i8** %arg0)
	ret i8* %0
}

define void @test_objc_moveWeak(i8** %arg0, i8** %arg1) {
; CHECK-LABEL: test_objc_moveWeak
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_moveWeak(i8** %arg0, i8** %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.moveWeak(i8** %arg0, i8** %arg1)
  ret void
}

define void @test_objc_release(i8* %arg0) {
; CHECK-LABEL: test_objc_release
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_release(i8* %arg0)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.release(i8* %arg0)
  ret void
}

define i8* @test_objc_retain(i8* %arg0) {
; CHECK-LABEL: test_objc_retain
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call i8* @objc_retain(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retain(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutorelease(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_retainAutorelease(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retainAutorelease(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutoreleaseReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleaseReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = tail call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutoreleasedReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleasedReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainBlock(i8* %arg0) {
; CHECK-LABEL: test_objc_retainBlock
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_retainBlock(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retainBlock(i8* %arg0)
	ret i8* %0
}

define void @test_objc_storeStrong(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_storeStrong
; CHECK-NEXT: entry
; CHECK-NEXT: call void @objc_storeStrong(i8** %arg0, i8* %arg1)
; CHECK-NEXT: ret void
entry:
  call void @llvm.objc.storeStrong(i8** %arg0, i8* %arg1)
	ret void
}

define i8* @test_objc_storeWeak(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_storeWeak
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_storeWeak(i8** %arg0, i8* %arg1)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.storeWeak(i8** %arg0, i8* %arg1)
	ret i8* %0
}

define i8* @test_objc_unsafeClaimAutoreleasedReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_unsafeClaimAutoreleasedReturnValue
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = tail call i8* @objc_unsafeClaimAutoreleasedReturnValue(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %arg0)
  ret i8* %0
}

define i8* @test_objc_retainedObject(i8* %arg0) {
; CHECK-LABEL: test_objc_retainedObject
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_retainedObject(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retainedObject(i8* %arg0)
  ret i8* %0
}

define i8* @test_objc_unretainedObject(i8* %arg0) {
; CHECK-LABEL: test_objc_unretainedObject
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_unretainedObject(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.unretainedObject(i8* %arg0)
  ret i8* %0
}

define i8* @test_objc_unretainedPointer(i8* %arg0) {
; CHECK-LABEL: test_objc_unretainedPointer
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_unretainedPointer(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.unretainedPointer(i8* %arg0)
  ret i8* %0
}

define i8* @test_objc_retain_autorelease(i8* %arg0) {
; CHECK-LABEL: test_objc_retain_autorelease
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i8* @objc_retain_autorelease(i8* %arg0)
; CHECK-NEXT: ret i8* %0
entry:
  %0 = call i8* @llvm.objc.retain.autorelease(i8* %arg0)
  ret i8* %0
}

define i32 @test_objc_sync_enter(i8* %arg0) {
; CHECK-LABEL: test_objc_sync_enter
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i32 @objc_sync_enter(i8* %arg0)
; CHECK-NEXT: ret i32 %0
entry:
  %0 = call i32 @llvm.objc.sync.enter(i8* %arg0)
  ret i32 %0
}

define i32 @test_objc_sync_exit(i8* %arg0) {
; CHECK-LABEL: test_objc_sync_exit
; CHECK-NEXT: entry
; CHECK-NEXT: %0 = call i32 @objc_sync_exit(i8* %arg0)
; CHECK-NEXT: ret i32 %0
entry:
  %0 = call i32 @llvm.objc.sync.exit(i8* %arg0)
  ret i32 %0
}

declare i8* @llvm.objc.autorelease(i8*)
declare void @llvm.objc.autoreleasePoolPop(i8*)
declare i8* @llvm.objc.autoreleasePoolPush()
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare void @llvm.objc.copyWeak(i8**, i8**)
declare void @llvm.objc.destroyWeak(i8**)
declare extern_weak i8* @llvm.objc.initWeak(i8**, i8*)
declare i8* @llvm.objc.loadWeak(i8**)
declare i8* @llvm.objc.loadWeakRetained(i8**)
declare void @llvm.objc.moveWeak(i8**, i8**)
declare void @llvm.objc.release(i8*)
declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.retainAutorelease(i8*)
declare i8* @llvm.objc.retainAutoreleaseReturnValue(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.retainBlock(i8*)
declare void @llvm.objc.storeStrong(i8**, i8*)
declare i8* @llvm.objc.storeWeak(i8**, i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.retainedObject(i8*)
declare i8* @llvm.objc.unretainedObject(i8*)
declare i8* @llvm.objc.unretainedPointer(i8*)
declare i8* @llvm.objc.retain.autorelease(i8*)
declare i32 @llvm.objc.sync.enter(i8*)
declare i32 @llvm.objc.sync.exit(i8*)

attributes #0 = { nounwind }

; CHECK: declare i8* @objc_autorelease(i8*)
; CHECK: declare void @objc_autoreleasePoolPop(i8*)
; CHECK: declare i8* @objc_autoreleasePoolPush()
; CHECK: declare i8* @objc_autoreleaseReturnValue(i8*)
; CHECK: declare void @objc_copyWeak(i8**, i8**)
; CHECK: declare void @objc_destroyWeak(i8**)
; CHECK: declare extern_weak i8* @objc_initWeak(i8**, i8*)
; CHECK: declare i8* @objc_loadWeak(i8**)
; CHECK: declare i8* @objc_loadWeakRetained(i8**)
; CHECK: declare void @objc_moveWeak(i8**, i8**)
; CHECK: declare void @objc_release(i8*) [[NLB:#[0-9]+]]
; CHECK: declare i8* @objc_retain(i8*) [[NLB]]
; CHECK: declare i8* @objc_retainAutorelease(i8*)
; CHECK: declare i8* @objc_retainAutoreleaseReturnValue(i8*)
; CHECK: declare i8* @objc_retainAutoreleasedReturnValue(i8*)
; CHECK: declare i8* @objc_retainBlock(i8*)
; CHECK: declare void @objc_storeStrong(i8**, i8*)
; CHECK: declare i8* @objc_storeWeak(i8**, i8*)
; CHECK: declare i8* @objc_unsafeClaimAutoreleasedReturnValue(i8*)
; CHECK: declare i8* @objc_retainedObject(i8*)
; CHECK: declare i8* @objc_unretainedObject(i8*)
; CHECK: declare i8* @objc_unretainedPointer(i8*)
; CHECK: declare i8* @objc_retain_autorelease(i8*)
; CHECK: declare i32 @objc_sync_enter(i8*)
; CHECK: declare i32 @objc_sync_exit(i8*)

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes [[NLB]] = { nonlazybind }
