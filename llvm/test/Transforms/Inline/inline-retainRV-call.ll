; RUN: opt < %s -inline -S | FileCheck %s

@g0 = global i8* null, align 8
declare i8* @foo0()

define i8* @callee0_autoreleaseRV() {
  %call = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  %1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call)
  ret i8* %call
}

; CHECK-LABEL: define void @test0_autoreleaseRV(
; CHECK: call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test0_autoreleaseRV() {
  %call = call i8* @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test0_claimRV_autoreleaseRV(
; CHECK: %[[CALL:.*]] = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void @llvm.objc.release(i8* %[[CALL]])
; CHECK-NEXT: ret void

define void @test0_claimRV_autoreleaseRV() {
  %call = call i8* @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1_autoreleaseRV(
; CHECK: invoke i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test1_autoreleaseRV() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i8* @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

; CHECK-LABEL: define void @test1_claimRV_autoreleaseRV(
; CHECK: %[[INVOKE:.*]] = invoke i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void @llvm.objc.release(i8* %[[INVOKE]])
; CHECK-NEXT: br

define void @test1_claimRV_autoreleaseRV() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i8* @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

define i8* @callee1_no_autoreleaseRV() {
  %call = call i8* @foo0()
  ret i8* %call
}

; CHECK-LABEL: define void @test2_no_autoreleaseRV(
; CHECK: call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test2_no_autoreleaseRV() {
  %call = call i8* @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test2_claimRV_no_autoreleaseRV(
; CHECK: call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test2_claimRV_no_autoreleaseRV() {
  %call = call i8* @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test3_no_autoreleaseRV(
; CHECK: invoke i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test3_no_autoreleaseRV() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i8* @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

define i8* @callee2_nocall() {
  %1 = load i8*, i8** @g0, align 8
  ret i8* %1
}

; Check that a call to @llvm.objc.retain is inserted if there is no matching
; autoreleaseRV call or a call.

; CHECK-LABEL: define void @test4_nocall(
; CHECK: %[[V0:.*]] = load i8*, i8** @g0,
; CHECK-NEXT: call i8* @llvm.objc.retain(i8* %[[V0]])
; CHECK-NEXT: ret void

define void @test4_nocall() {
  %call = call i8* @callee2_nocall() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test4_claimRV_nocall(
; CHECK: %[[V0:.*]] = load i8*, i8** @g0,
; CHECK-NEXT: ret void

define void @test4_claimRV_nocall() {
  %call = call i8* @callee2_nocall() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; Check that a call to @llvm.objc.retain is inserted if call to @foo already has
; the attribute. I'm not sure this will happen in practice.

define i8* @callee3_marker() {
  %1 = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret i8* %1
}

; CHECK-LABEL: define void @test5(
; CHECK: %[[V0:.*]] = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: call i8* @llvm.objc.retain(i8* %[[V0]])
; CHECK-NEXT: ret void

define void @test5() {
  %call = call i8* @callee3_marker() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; Don't pair up an autoreleaseRV in the callee and an retainRV in the caller
; if there is an instruction between the ret instruction and the call to
; autoreleaseRV that isn't a cast instruction.

define i8* @callee0_autoreleaseRV2() {
  %call = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  %1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call)
  store i8* null, i8** @g0
  ret i8* %call
}

; CHECK-LABEL: define void @test6(
; CHECK: %[[V0:.*]] = call i8* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[V0]])
; CHECK: store i8* null, i8** @g0, align 8
; CHECK: call i8* @llvm.objc.retain(i8* %[[V0]])
; CHECK-NEXT: ret void

define void @test6() {
  %call = call i8* @callee0_autoreleaseRV2() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i32 @__gxx_personality_v0(...)
