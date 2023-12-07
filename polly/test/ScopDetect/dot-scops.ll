; RUN: opt %loadPolly -polly-scops -dot-scops -disable-output < %s
;
; Check that the ScopPrinter does not crash.
; ScopPrinter needs the ScopDetection pass, which should depend on
; ScalarEvolution transitively.
;
; FIXME: -dot-scops always prints to the same hardcoded filename
;        scops.<functionname>.dot. If there is another test with the same
;        function name and printing a dot file there will be a race condition
;        when running tests in parallel.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i32 %n, i32 %m, ptr noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %inner.for, label %outer.exit

  inner.for:
    %i = phi i32 [1, %outer.for], [%i.inc, %inner.inc]
    %b = phi double [0.0, %outer.for], [%a, %inner.inc]
    %i.cmp = icmp slt i32 %i, %m
    br i1 %i.cmp, label %body1, label %inner.exit

    body1:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %i
      %a = load double, ptr %A_idx
      store double %a, ptr %A_idx
      br label %inner.inc

  inner.inc:
    %i.inc = add nuw nsw i32 %i, 1
    br label %inner.for

  inner.exit:
    br label %outer.inc

outer.inc:
  store double %b, ptr %A
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}
