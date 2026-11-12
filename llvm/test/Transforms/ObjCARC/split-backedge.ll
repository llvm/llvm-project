; RUN: opt -S -passes=objc-arc < %s | FileCheck %s

; Handle a retain+release pair entirely contained within a split loop backedge.
; rdar://11256239

; CHECK-LABEL: define void @test0(
; CHECK: call ptr @llvm.objc.retain(ptr %call) [[NUW:#[0-9]+]]
; CHECK: call ptr @llvm.objc.retain(ptr %call) [[NUW]]
; CHECK: call ptr @llvm.objc.retain(ptr %cond) [[NUW]]
; CHECK: call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK: call void @llvm.objc.release(ptr %call) [[NUW]]
; CHECK: call void @llvm.objc.release(ptr %cond) [[NUW]]
define void @test0() personality ptr @__objc_personality_v0 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.cond
  %call = invoke ptr @returner()
          to label %invoke.cont unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

invoke.cont:                                      ; preds = %while.body
  %t0 = call ptr @llvm.objc.retain(ptr %call) nounwind
  %t1 = call ptr @llvm.objc.retain(ptr %call) nounwind
  %call.i1 = invoke ptr @returner()
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:                                     ; preds = %invoke.cont
  %cond = select i1 undef, ptr null, ptr %call
  %t2 = call ptr @llvm.objc.retain(ptr %cond) nounwind
  call void @llvm.objc.release(ptr %call) nounwind
  call void @llvm.objc.release(ptr %call) nounwind
  call void @use_pointer(ptr %cond)
  call void @llvm.objc.release(ptr %cond) nounwind
  br label %while.body

lpad:                                             ; preds = %invoke.cont, %while.body
  %t4 = landingpad { ptr, i32 }
          catch ptr null
  ret void
}

declare ptr @returner()
declare i32 @__objc_personality_v0(...)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.retain(ptr)
declare void @use_pointer(ptr)

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
