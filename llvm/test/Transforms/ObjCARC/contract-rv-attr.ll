; RUN: opt -objc-arc-contract -S < %s | FileCheck %s
; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; CHECK-LABEL: define void @test0() {
; CHECK: %[[CALL:.*]] = notail call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NOT: call i8* @llvm.objc.retainAutoreleasedReturnValue(

define void @test0() {
  %call1 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1() {
; CHECK: %[[CALL:.*]] = notail call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NOT: call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(

define void @test1() {
  %call1 = call i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL:define i8* @test2(
; CHECK: %[[V0:.*]] = invoke i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]

; CHECK-NOT: = call i8* @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: br

; CHECK: %[[V2:.*]] = invoke i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]

; CHECK-NOT: = call i8* @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: br

; CHECK: %[[RETVAL:.*]] = phi i8* [ %[[V0]], {{.*}} ], [ %[[V2]], {{.*}} ]
; CHECK: ret i8* %[[RETVAL]]

define i8* @test2(i1 zeroext %b) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 %b, label %if.then, label %if.end

if.then:
  %call1 = invoke i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %cleanup unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

if.end:
  %call3 = invoke i8* @foo() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %cleanup unwind label %lpad

cleanup:
  %retval.0 = phi i8* [ %call1, %if.then ], [ %call3, %if.end ]
  ret i8* %retval.0
}

; "clang.arc.attachedcall" is ignored if the return type of the called function is void.
; CHECK-LABEL: define void @test3(
; CHECK: call void @foo2() #[[ATTR1:.*]] [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test3() {
  call void @foo2() #0 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

declare i8* @foo()
declare void @foo2()
declare i32 @__gxx_personality_v0(...)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)

!llvm.module.flags = !{!0}

; CHECK: attributes #[[ATTR1]] = { noreturn }
attributes #0 = { noreturn }

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue"}
