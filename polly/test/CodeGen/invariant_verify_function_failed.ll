; RUN: opt %loadPolly -polly-detect -polly-codegen -polly-invariant-load-hoisting=true -analyze < %s | FileCheck %s
;
; This crashed at some point as the pointer returned by the call
; to @__errno_location is invariant and defined in the SCoP but not
; loaded. Therefore it is not hoisted and consequently not available
; at the beginning of the SCoP where we would need it if we would try
; to hoist %tmp. We don't try to hoist %tmp anymore but this test still
; checks that this passes to code generation and produces valid code.
;
; This SCoP is currently rejected because %call9 is not considered a valid
; base pointer.
;
; CHECK-NOT: Valid Region for Scop
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @fileblobSetFilename() #0 {
entry:
  br i1 undef, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  br i1 undef, label %land.lhs.true, label %if.end.18

land.lhs.true:                                    ; preds = %if.end
  %call9 = tail call i32* @__errno_location() #2
  %tmp = load i32, i32* %call9, align 4, !tbaa !1
  br i1 false, label %if.then.12, label %if.end.18

if.then.12:                                       ; preds = %land.lhs.true
  br label %if.end.18

if.end.18:                                        ; preds = %if.then.12, %land.lhs.true, %if.end
  %fd.0 = phi i32 [ undef, %if.then.12 ], [ undef, %land.lhs.true ], [ undef, %if.end ]
  br i1 undef, label %if.then.21, label %if.end.27

if.then.21:                                       ; preds = %if.end.18
  unreachable

if.end.27:                                        ; preds = %if.end.18
  br label %cleanup

cleanup:                                          ; preds = %if.end.27, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare i32* @__errno_location() #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
