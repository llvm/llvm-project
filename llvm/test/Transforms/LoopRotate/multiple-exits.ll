; RUN: opt -S -passes=loop-rotate < %s -verify-loop-info -verify-dom-info -verify-memoryssa | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; PR7447
define i32 @test1(ptr nocapture %a) nounwind readonly {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond1, %entry
  %sum.0 = phi i32 [ 0, %entry ], [ %sum.1, %for.cond1 ]
  %i.0 = phi i1 [ true, %entry ], [ false, %for.cond1 ]
  br i1 %i.0, label %for.cond1, label %return

for.cond1:                                        ; preds = %for.cond, %land.rhs
  %sum.1 = phi i32 [ %add, %land.rhs ], [ %sum.0, %for.cond ]
  %i.1 = phi i32 [ %inc, %land.rhs ], [ 0, %for.cond ]
  %cmp2 = icmp ult i32 %i.1, 100
  br i1 %cmp2, label %land.rhs, label %for.cond

land.rhs:                                         ; preds = %for.cond1
  %conv = zext i32 %i.1 to i64
  %arrayidx = getelementptr inbounds [100 x i32], ptr %a, i64 0, i64 %conv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum.1
  %cmp4 = icmp ugt i32 %add, 1000
  %inc = add i32 %i.1, 1
  br i1 %cmp4, label %return, label %for.cond1

return:                                           ; preds = %for.cond, %land.rhs
  %retval.0 = phi i32 [ 1000, %land.rhs ], [ %sum.0, %for.cond ]
  ret i32 %retval.0

; CHECK-LABEL: @test1(
; CHECK: for.cond1.preheader:
; CHECK: %sum.04 = phi i32 [ 0, %entry ], [ %sum.1.lcssa, %for.cond.loopexit ]
; CHECK: br label %for.cond1

; CHECK: for.cond1:
; CHECK: %sum.1 = phi i32 [ %add, %land.rhs ], [ %sum.04, %for.cond1.preheader ]
; CHECK: %i.1 = phi i32 [ %inc, %land.rhs ], [ 0, %for.cond1.preheader ]
; CHECK: %cmp2 = icmp ult i32 %i.1, 100
; CHECK: br i1 %cmp2, label %land.rhs, label %for.cond.loopexit
}

define void @test2(i32 %x) nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cmp = icmp eq i32 %i.0, %x
  br i1 %cmp, label %return.loopexit, label %for.body

for.body:                                         ; preds = %for.cond
  %call = tail call i32 @foo(i32 %i.0) nounwind
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.end, label %a

if.end:                                           ; preds = %for.body
  %call1 = tail call i32 @foo(i32 42) nounwind
  %inc = add i32 %i.0, 1
  br label %for.cond

a:                                                ; preds = %for.body
  %call2 = tail call i32 @bar(i32 1) nounwind
  br label %return

return.loopexit:                                  ; preds = %for.cond
  br label %return

return:                                           ; preds = %return.loopexit, %a
  ret void

; CHECK-LABEL: @test2(
; CHECK: if.end:
; CHECK: %inc = add i32 %i.02, 1
; CHECK: %cmp = icmp eq i32 %inc, %x
; CHECK: br i1 %cmp, label %for.cond.return.loopexit_crit_edge, label %for.body
}

declare i32 @foo(i32)

declare i32 @bar(i32)

@_ZTIi = external constant ptr

; Verify dominators.
define void @test3(i32 %x) personality ptr @__gxx_personality_v0 {
entry:
  %cmp2 = icmp eq i32 0, %x
  br i1 %cmp2, label %try.cont.loopexit, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.03 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  invoke void @_Z3fooi(i32 %i.03)
          to label %for.inc unwind label %lpad

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.03, 1
  %cmp = icmp eq i32 %inc, %x
  br i1 %cmp, label %for.cond.try.cont.loopexit_crit_edge, label %for.body

lpad:                                             ; preds = %for.body
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %4 = tail call ptr @__cxa_begin_catch(ptr %1) nounwind
  br i1 true, label %invoke.cont2.loopexit, label %for.body.i.lr.ph

for.body.i.lr.ph:                                 ; preds = %catch
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i.lr.ph, %for.inc.i
  %i.0.i1 = phi i32 [ 0, %for.body.i.lr.ph ], [ %inc.i, %for.inc.i ]
  invoke void @_Z3fooi(i32 %i.0.i1)
          to label %for.inc.i unwind label %lpad.i

for.inc.i:                                        ; preds = %for.body.i
  %inc.i = add i32 %i.0.i1, 1
  %cmp.i = icmp eq i32 %inc.i, 0
  br i1 %cmp.i, label %for.cond.i.invoke.cont2.loopexit_crit_edge, label %for.body.i

lpad.i:                                           ; preds = %for.body.i
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = extractvalue { ptr, i32 } %5, 1
  %matches.i = icmp eq i32 %7, %3
  br i1 %matches.i, label %catch.i, label %lpad1.body

catch.i:                                          ; preds = %lpad.i
  %8 = tail call ptr @__cxa_begin_catch(ptr %6) nounwind
  invoke void @test3(i32 0)
          to label %invoke.cont2.i unwind label %lpad1.i

invoke.cont2.i:                                   ; preds = %catch.i
  tail call void @__cxa_end_catch() nounwind
  br label %invoke.cont2

lpad1.i:                                          ; preds = %catch.i
  %9 = landingpad { ptr, i32 }
          cleanup
  %10 = extractvalue { ptr, i32 } %9, 0
  %11 = extractvalue { ptr, i32 } %9, 1
  tail call void @__cxa_end_catch() nounwind
  br label %lpad1.body

for.cond.i.invoke.cont2.loopexit_crit_edge:       ; preds = %for.inc.i
  br label %invoke.cont2.loopexit

invoke.cont2.loopexit:                            ; preds = %for.cond.i.invoke.cont2.loopexit_crit_edge, %catch
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %invoke.cont2.loopexit, %invoke.cont2.i
  tail call void @__cxa_end_catch() nounwind
  br label %try.cont

for.cond.try.cont.loopexit_crit_edge:             ; preds = %for.inc
  br label %try.cont.loopexit

try.cont.loopexit:                                ; preds = %for.cond.try.cont.loopexit_crit_edge, %entry
  br label %try.cont

try.cont:                                         ; preds = %try.cont.loopexit, %invoke.cont2
  ret void

lpad1.body:                                       ; preds = %lpad1.i, %lpad.i
  %exn.slot.0.i = phi ptr [ %10, %lpad1.i ], [ %6, %lpad.i ]
  %ehselector.slot.0.i = phi i32 [ %11, %lpad1.i ], [ %7, %lpad.i ]
  tail call void @__cxa_end_catch() nounwind
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1.body, %lpad
  %exn.slot.0 = phi ptr [ %exn.slot.0.i, %lpad1.body ], [ %1, %lpad ]
  %ehselector.slot.0 = phi i32 [ %ehselector.slot.0.i, %lpad1.body ], [ %2, %lpad ]
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val5 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val5
}

declare void @_Z3fooi(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

define void @test4(i1 %arg) nounwind uwtable {
entry:
  br label %"7"

"3":                                              ; preds = %"7"
  br i1 %arg, label %"31", label %"4"

"4":                                              ; preds = %"3"
  %. = select i1 undef, float 0x3F50624DE0000000, float undef
  %0 = add i32 %1, 1
  br label %"7"

"7":                                              ; preds = %"4", %entry
  %1 = phi i32 [ %0, %"4" ], [ 0, %entry ]
  %2 = icmp slt i32 %1, 100
  br i1 %2, label %"3", label %"8"

"8":                                              ; preds = %"7"
  br i1 %arg, label %"9", label %"31"

"9":                                              ; preds = %"8"
  br label %"33"

"27":                                             ; preds = %"31"
  unreachable

"31":                                             ; preds = %"8", %"3"
  br i1 %arg, label %"27", label %"32"

"32":                                             ; preds = %"31"
  br label %"33"

"33":                                             ; preds = %"32", %"9"
  ret void
}
