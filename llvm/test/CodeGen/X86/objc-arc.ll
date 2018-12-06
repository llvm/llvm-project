; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; Make sure calls to the objc intrinsics are translated to calls in to the
; runtime

define i8* @test_objc_autorelease(i8* %arg0) {
; CHECK-LABEL: test_objc_autorelease
; CHECK: callq _objc_autorelease
entry:
    %0 = call i8* @llvm.objc.autorelease(i8* %arg0)
	ret i8* %0
}

define void @test_objc_autoreleasePoolPop(i8* %arg0) {
; CHECK-LABEL: test_objc_autoreleasePoolPop
; CHECK: callq _objc_autoreleasePoolPop
entry:
    call void @llvm.objc.autoreleasePoolPop(i8* %arg0)
    ret void
}

define i8* @test_objc_autoreleasePoolPush() {
; CHECK-LABEL: test_objc_autoreleasePoolPush
; CHECK: callq _objc_autoreleasePoolPush
entry:
    %0 = call i8* @llvm.objc.autoreleasePoolPush()
	ret i8* %0
}

define i8* @test_objc_autoreleaseReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_autoreleaseReturnValue
; CHECK: callq _objc_autoreleaseReturnValue
entry:
    %0 = call i8* @llvm.objc.autoreleaseReturnValue(i8* %arg0)
	ret i8* %0
}

define void @test_objc_copyWeak(i8** %arg0, i8** %arg1) {
; CHECK-LABEL: test_objc_copyWeak
; CHECK: callq _objc_copyWeak
entry:
    call void @llvm.objc.copyWeak(i8** %arg0, i8** %arg1)
    ret void
}

define void @test_objc_destroyWeak(i8** %arg0) {
; CHECK-LABEL: test_objc_destroyWeak
; CHECK: callq _objc_destroyWeak
entry:
    call void @llvm.objc.destroyWeak(i8** %arg0)
    ret void
}

define i8* @test_objc_initWeak(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_initWeak
; CHECK: callq _objc_initWeak
entry:
    %0 = call i8* @llvm.objc.initWeak(i8** %arg0, i8* %arg1)
	ret i8* %0
}

define i8* @test_objc_loadWeak(i8** %arg0) {
; CHECK-LABEL: test_objc_loadWeak
; CHECK: callq _objc_loadWeak
entry:
    %0 = call i8* @llvm.objc.loadWeak(i8** %arg0)
	ret i8* %0
}

define i8* @test_objc_loadWeakRetained(i8** %arg0) {
; CHECK-LABEL: test_objc_loadWeakRetained
; CHECK: callq _objc_loadWeakRetained
entry:
    %0 = call i8* @llvm.objc.loadWeakRetained(i8** %arg0)
	ret i8* %0
}

define void @test_objc_moveWeak(i8** %arg0, i8** %arg1) {
; CHECK-LABEL: test_objc_moveWeak
; CHECK: callq _objc_moveWeak
entry:
    call void @llvm.objc.moveWeak(i8** %arg0, i8** %arg1)
    ret void
}

define void @test_objc_release(i8* %arg0) {
; CHECK-LABEL: test_objc_release
; CHECK: callq _objc_release
entry:
    call void @llvm.objc.release(i8* %arg0)
    ret void
}

define i8* @test_objc_retain(i8* %arg0) {
; CHECK-LABEL: test_objc_retain
; CHECK: callq _objc_retain
entry:
    %0 = call i8* @llvm.objc.retain(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutorelease(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutorelease
; CHECK: callq _objc_retainAutorelease
entry:
    %0 = call i8* @llvm.objc.retainAutorelease(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutoreleaseReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleaseReturnValue
; CHECK: callq _objc_retainAutoreleaseReturnValue
entry:
    %0 = call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainAutoreleasedReturnValue(i8* %arg0) {
; CHECK-LABEL: test_objc_retainAutoreleasedReturnValue
; CHECK: callq _objc_retainAutoreleasedReturnValue
entry:
    %0 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %arg0)
	ret i8* %0
}

define i8* @test_objc_retainBlock(i8* %arg0) {
; CHECK-LABEL: test_objc_retainBlock
; CHECK: callq _objc_retainBlock
entry:
    %0 = call i8* @llvm.objc.retainBlock(i8* %arg0)
	ret i8* %0
}

define void @test_objc_storeStrong(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_storeStrong
; CHECK: callq _objc_storeStrong
entry:
    call void @llvm.objc.storeStrong(i8** %arg0, i8* %arg1)
	ret void
}

define i8* @test_objc_storeWeak(i8** %arg0, i8* %arg1) {
; CHECK-LABEL: test_objc_storeWeak
; CHECK: callq _objc_storeWeak
entry:
    %0 = call i8* @llvm.objc.storeWeak(i8** %arg0, i8* %arg1)
	ret i8* %0
}

declare i8* @llvm.objc.autorelease(i8*)
declare void @llvm.objc.autoreleasePoolPop(i8*)
declare i8* @llvm.objc.autoreleasePoolPush()
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare void @llvm.objc.copyWeak(i8**, i8**)
declare void @llvm.objc.destroyWeak(i8**)
declare i8* @llvm.objc.initWeak(i8**, i8*)
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
