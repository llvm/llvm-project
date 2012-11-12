; RUN: opt -S -objc-arc < %s | FileCheck %s

; Handle a retain+release pair entirely contained within a split loop backedge.
; rdar://11256239

; CHECK: define void @test0
; CHECK: call i8* @objc_retain(i8* %call) nounwind
; CHECK: call i8* @objc_retain(i8* %call) nounwind
; CHECK: call i8* @objc_retain(i8* %cond) nounwind
; CHECK: call void @objc_release(i8* %call) nounwind
; CHECK: call void @objc_release(i8* %call) nounwind
; CHECK: call void @objc_release(i8* %cond) nounwind
define void @test0() {
entry:
  br label %while.body

while.body:                                       ; preds = %while.cond
  %call = invoke i8* @returner()
          to label %invoke.cont unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

invoke.cont:                                      ; preds = %while.body
  %t0 = call i8* @objc_retain(i8* %call) nounwind
  %t1 = call i8* @objc_retain(i8* %call) nounwind
  %call.i1 = invoke i8* @returner()
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:                                     ; preds = %invoke.cont
  %cond = select i1 undef, i8* null, i8* %call
  %t2 = call i8* @objc_retain(i8* %cond) nounwind
  call void @objc_release(i8* %call) nounwind
  call void @objc_release(i8* %call) nounwind
  call void @use_pointer(i8* %cond)
  call void @objc_release(i8* %cond) nounwind
  br label %while.body

lpad:                                             ; preds = %invoke.cont, %while.body
  %t4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null
  ret void
}

declare i8* @returner()
declare i32 @__objc_personality_v0(...)
declare void @objc_release(i8*)
declare i8* @objc_retain(i8*)
declare void @use_pointer(i8*)

!0 = metadata !{}
