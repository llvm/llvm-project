; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local i8* @bar(i32 %arg) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb5

bb1:                                              ; preds = %bb
  %call26 = invoke i8* @invoke_callee() #2
          to label %cont unwind label %lpad
lpad:                                            ; preds = %if.end
  %0 = landingpad { i8*, i32 }
         cleanup
  resume { i8*, i32 } undef

cont:
    br label %bb5

bb5:                                              ; preds = %bb4, %bb1, %bb
  %retval = phi i8* [ %call26, %cont ], [ undef, %bb]
  ret i8* %retval
}

; CHECK-LABEL: @dummy_caller
; CHECK-LABEL: bb:
; CHECK-NEXT:  [[CALL26LOC:%.*]] = alloca i8*
; CHECK-LABEL: codeRepl.i:
; CHECK-NEXT:   %lt.cast.i = bitcast i8** [[CALL26LOC]] to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 -1, i8* %lt.cast.i)
; CHECK-NEXT:   call void @bar.1.bb1(i8** [[CALL26LOC]])
; CHECK-NEXT:   %call26.reload.i = load i8*, i8** [[CALL26LOC]]
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 -1, i8* %lt.cast.i)
define i8* @dummy_caller(i32 %arg) {
bb:
  %tmp = tail call i8* @bar(i32 %arg)
  ret i8* %tmp
}

; CHECK-LABEL: define internal void @bar.1.bb1
; CHECK-LABEL: bb1:
; CHECK-NEXT:    %call26 = invoke i8* @invoke_callee()
; CHECK-NEXT:            to label %cont unwind label %lpad
; CHECK-LABEL: cont:
; CHECK-NEXT:    store i8* %call26, i8** %call26.out
; CHECK-NEXT:    br label %bb5.exitStub

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @invoke_callee() local_unnamed_addr #1

declare dso_local i32 @__gxx_personality_v0(...)
