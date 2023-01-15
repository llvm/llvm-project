; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; CHECK-LABEL: define void @test0() {
; CHECK: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test0() {
  %call1 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1() {
; CHECK: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test1() {
  %call1 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

declare ptr @foo()
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
