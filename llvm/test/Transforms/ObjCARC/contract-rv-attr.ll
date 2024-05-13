; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; CHECK-LABEL: define void @test0() {
; CHECK: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NOT: call ptr @llvm.objc.retainAutoreleasedReturnValue(

define void @test0() {
  %call1 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1() {
; CHECK: %[[CALL:.*]] = notail call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NOT: call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(

define void @test1() {
  %call1 = call ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL:define ptr @test2(
; CHECK: %[[V0:.*]] = invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

; CHECK-NOT: = call ptr @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: br

; CHECK: %[[V2:.*]] = invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

; CHECK-NOT: = call ptr @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: br

; CHECK: %[[RETVAL:.*]] = phi ptr [ %[[V0]], {{.*}} ], [ %[[V2]], {{.*}} ]
; CHECK: ret ptr %[[RETVAL]]

define ptr @test2(i1 zeroext %b) personality ptr @__gxx_personality_v0 {
entry:
  br i1 %b, label %if.then, label %if.end

if.then:
  %call1 = invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %cleanup unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef

if.end:
  %call3 = invoke ptr @foo() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %cleanup unwind label %lpad

cleanup:
  %retval.0 = phi ptr [ %call1, %if.then ], [ %call3, %if.end ]
  ret ptr %retval.0
}

; "clang.arc.attachedcall" is ignored if the return type of the called function is void.
; CHECK-LABEL: define void @test3(
; CHECK: call void @foo2() #[[ATTR1:.*]] [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test3() {
  call void @foo2() #0 [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

declare ptr @foo()
declare void @foo2()
declare i32 @__gxx_personality_v0(...)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)

!llvm.module.flags = !{!0}

; CHECK: attributes #[[ATTR1]] = { noreturn }
attributes #0 = { noreturn }

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue"}
