; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local ptr @bar(i32 %arg) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb5

bb1:                                              ; preds = %bb
  %call26 = invoke ptr @invoke_callee() #2
          to label %cont unwind label %lpad
lpad:                                            ; preds = %if.end
  %0 = landingpad { ptr, i32 }
         cleanup
  resume { ptr, i32 } undef

cont:
    br label %bb5

bb5:                                              ; preds = %bb4, %bb1, %bb
  %retval = phi ptr [ %call26, %cont ], [ undef, %bb]
  ret ptr %retval
}

; CHECK-LABEL: @dummy_caller
; CHECK-LABEL: bb:
; CHECK-LABEL: codeRepl.i:
; CHECK-NEXT:   call ptr @bar.1.bb1()
define ptr @dummy_caller(i32 %arg) {
bb:
  %tmp = tail call ptr @bar(i32 %arg)
  ret ptr %tmp
}

; CHECK-LABEL: define internal ptr @bar.1.bb1
; CHECK-LABEL: bb1:
; CHECK-NEXT:    %call26 = invoke ptr @invoke_callee()
; CHECK-NEXT:            to label %cont unwind label %lpad
; CHECK-LABEL: cont:
; CHECK-NEXT:    br label %bb5.exitStub
; CHECK-LABEL: bb5.exitStub:
; CHECK-NEXT:    ret ptr %call26

; Function Attrs: nobuiltin
declare dso_local noalias nonnull ptr @invoke_callee() local_unnamed_addr #1

declare dso_local i32 @__gxx_personality_v0(...)
