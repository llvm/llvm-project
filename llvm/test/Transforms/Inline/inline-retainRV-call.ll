; RUN: opt < %s -passes=inline -S | FileCheck %s

@g0 = global ptr null, align 8
declare ptr @foo0()

define ptr @callee0_autoreleaseRV() {
  %call = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  %1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call)
  ret ptr %call
}

; CHECK-LABEL: define void @test0_autoreleaseRV(
; CHECK: call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test0_autoreleaseRV() {
  %call = call ptr @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test0_claimRV_autoreleaseRV(
; CHECK: %[[CALL:.*]] = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void @llvm.objc.release(ptr %[[CALL]])
; CHECK-NEXT: ret void

define void @test0_claimRV_autoreleaseRV() {
  %call = call ptr @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test1_autoreleaseRV(
; CHECK: invoke ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test1_autoreleaseRV() personality ptr @__gxx_personality_v0 {
entry:
  %call = invoke ptr @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef
}

; CHECK-LABEL: define void @test1_claimRV_autoreleaseRV(
; CHECK: %[[INVOKE:.*]] = invoke ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call void @llvm.objc.release(ptr %[[INVOKE]])
; CHECK-NEXT: br

define void @test1_claimRV_autoreleaseRV() personality ptr @__gxx_personality_v0 {
entry:
  %call = invoke ptr @callee0_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef
}

define ptr @callee1_no_autoreleaseRV() {
  %call = call ptr @foo0()
  ret ptr %call
}

; CHECK-LABEL: define void @test2_no_autoreleaseRV(
; CHECK: call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test2_no_autoreleaseRV() {
  %call = call ptr @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test2_claimRV_no_autoreleaseRV(
; CHECK: call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
; CHECK-NEXT: ret void

define void @test2_claimRV_no_autoreleaseRV() {
  %call = call ptr @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test3_no_autoreleaseRV(
; CHECK: invoke ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]

define void @test3_no_autoreleaseRV() personality ptr @__gxx_personality_v0 {
entry:
  %call = invoke ptr @callee1_no_autoreleaseRV() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef
}

define ptr @callee2_nocall() {
  %1 = load ptr, ptr @g0, align 8
  ret ptr %1
}

; Check that a call to @llvm.objc.retain is inserted if there is no matching
; autoreleaseRV call or a call.

; CHECK-LABEL: define void @test4_nocall(
; CHECK: %[[V0:.*]] = load ptr, ptr @g0,
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %[[V0]])
; CHECK-NEXT: ret void

define void @test4_nocall() {
  %call = call ptr @callee2_nocall() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; CHECK-LABEL: define void @test4_claimRV_nocall(
; CHECK: %[[V0:.*]] = load ptr, ptr @g0,
; CHECK-NEXT: ret void

define void @test4_claimRV_nocall() {
  %call = call ptr @callee2_nocall() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  ret void
}

; Check that a call to @llvm.objc.retain is inserted if call to @foo already has
; the attribute. I'm not sure this will happen in practice.

define ptr @callee3_marker() {
  %1 = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret ptr %1
}

; CHECK-LABEL: define void @test5(
; CHECK: %[[V0:.*]] = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %[[V0]])
; CHECK-NEXT: ret void

define void @test5() {
  %call = call ptr @callee3_marker() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

; Don't pair up an autoreleaseRV in the callee and an retainRV in the caller
; if there is an instruction between the ret instruction and the call to
; autoreleaseRV that isn't a cast instruction.

define ptr @callee0_autoreleaseRV2() {
  %call = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  %1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call)
  store ptr null, ptr @g0
  ret ptr %call
}

; CHECK-LABEL: define void @test6(
; CHECK: %[[V0:.*]] = call ptr @foo0() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK: call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[V0]])
; CHECK: store ptr null, ptr @g0, align 8
; CHECK: call ptr @llvm.objc.retain(ptr %[[V0]])
; CHECK-NEXT: ret void

define void @test6() {
  %call = call ptr @callee0_autoreleaseRV2() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  ret void
}

declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare i32 @__gxx_personality_v0(...)
