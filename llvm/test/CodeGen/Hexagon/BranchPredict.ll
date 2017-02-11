; RUN: llc -march=hexagon -ifcvt-limit=0 < %s | FileCheck %s

; Check if the branch probabilities are reflected in the instructions:
; The basic block placement pass should place the more probable successor
; block as the fall-through block. The unconditional jump in the predecessor
; should then get the right hint (not_taken or ":nt")


@j = external global i32

define i32 @foo(i32 %a) nounwind {
; CHECK: if (!p{{[0-3]}}.new) jump:nt
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else, label %if.then, !prof !0

if.then:                                          ; preds = %entry
  %add = add nsw i32 %a, 10
  %call = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 %add) nounwind
  br label %return

if.else:                                          ; preds = %entry
  %call2 = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 4) nounwind
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %retval.0
}

declare i32 @foobar(...)

define i32 @bar(i32 %a) nounwind {
; CHECK: if (p{{[0-3]}}.new) jump:nt
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else, label %if.then, !prof !1

if.then:                                          ; preds = %entry
  %add = add nsw i32 %a, 10
  %call = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 %add) nounwind
  br label %return

if.else:                                          ; preds = %entry
  %call2 = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 4) nounwind
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %retval.0
}

define i32 @foo_bar(i32 %a, i16 signext %b) nounwind {
; CHECK: if (!cmp.eq(r{{[0-9]*}}.new,#0)) jump:nt
entry:
  %0 = load i32, i32* @j, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.else, label %if.then, !prof !0

if.then:                                          ; preds = %entry
  %add = add nsw i32 %a, 10
  %call = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 %add) nounwind
  br label %return

if.else:                                          ; preds = %entry
  %add1 = add nsw i32 %a, 4
  %call2 = tail call i32 bitcast (i32 (...)* @foobar to i32 (i32)*)(i32 %add1) nounwind
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call2, %if.else ]
  ret i32 %retval.0
}

!0 = !{!"branch_weights", i32 64, i32 4}
!1 = !{!"branch_weights", i32 4, i32 64}
