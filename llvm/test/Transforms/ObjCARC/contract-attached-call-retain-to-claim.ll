; RUN: opt -passes=objc-arc-contract -arc-contract-use-objc-claim-rv=1 -S < %s | FileCheck %s --check-prefixes=CHECK,CLAIM
; RUN: opt -passes=objc-arc-contract -arc-contract-use-objc-claim-rv=0 -S < %s | FileCheck %s --check-prefixes=CHECK,RETAIN

; CHECK-LABEL: define void @test0() {
; CLAIM: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.claimAutoreleasedReturnValue) ]
; RETAIN: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
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

; CHECK-LABEL: define void @test2() {
; CLAIM: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.claimAutoreleasedReturnValue), "otherbundle"() ]
; RETAIN: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue), "otherbundle"() ]
; CHECK-NEXT: ret void

define void @test2() {
  %call1 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue), "otherbundle"() ]
  ret void
}

declare ptr @foo()
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
