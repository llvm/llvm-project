; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s
; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; CHECK-LABEL: define void @test0() {
; CHECK: %[[CALL:.*]] = notail call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test0() {
  %call1 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1() {
; CHECK: %[[CALL:.*]] = notail call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test1() {
  %call1 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

declare i8* @foo()
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
