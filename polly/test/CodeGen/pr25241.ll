; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; PR25241 (https://llvm.org/bugs/show_bug.cgi?id=25241)
; Ensure that synthesized values of a PHI node argument are generated in the
; incoming block, not in the PHI's block.

; CHECK-LABEL: polly.stmt.if.then.862:
; CHECK:         %[[R1:[0-9]+]] = add i32 %tmp, 1
; CHECK:         br label

; CHECK-LABEL: polly.stmt.while.body.740.region_exiting:
; CHECK:         %polly.curr.3 = phi i32 [ %[[R1]], %polly.stmt.if.then.862 ], [ undef, %polly.stmt.if.else.864 ]
; CHECK:         br label %polly.stmt.polly.merge_new_and_old.exit

; CHECK-LABEL: polly.stmt.polly.merge_new_and_old.exit:
; CHECK:         store i32 %polly.curr.3, i32* %curr.3.s2a
; CHECK:         br label %polly.exiting

; CHECK-LABEL: polly.exiting:
; CHECK:         %curr.3.ph.final_reload = load i32, i32* %curr.3.s2a
; CHECK:         br label


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @BZ2_decompress() #0 {
entry:
  %tmp = load i32, i32* undef, align 4, !tbaa !1
  switch i32 undef, label %save_state_and_return [
    i32 34, label %sw.bb.748
    i32 35, label %if.then.813
  ]

while.body.740:                                   ; preds = %if.else.864, %if.then.862
  %curr.3 = phi i32 [ %inc863, %if.then.862 ], [ undef, %if.else.864 ]
  ret void

sw.bb.748:                                        ; preds = %entry
  ret void

if.then.813:                                      ; preds = %entry
  %conv823903 = and i32 undef, undef
  %cmp860 = icmp eq i32 %conv823903, 0
  br i1 %cmp860, label %if.then.862, label %if.else.864

if.then.862:                                      ; preds = %if.then.813
  %inc863 = add nsw i32 %tmp, 1
  br label %while.body.740

if.else.864:                                      ; preds = %if.then.813
  br label %while.body.740

save_state_and_return:                            ; preds = %entry
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
!1 = !{!2, !6, i64 64092}
!2 = !{!"", !3, i64 0, !6, i64 8, !4, i64 12, !6, i64 16, !4, i64 20, !6, i64 24, !6, i64 28, !6, i64 32, !6, i64 36, !6, i64 40, !4, i64 44, !6, i64 48, !6, i64 52, !6, i64 56, !6, i64 60, !6, i64 64, !4, i64 68, !6, i64 1092, !4, i64 1096, !4, i64 2124, !3, i64 3152, !3, i64 3160, !3, i64 3168, !6, i64 3176, !6, i64 3180, !6, i64 3184, !6, i64 3188, !6, i64 3192, !4, i64 3196, !4, i64 3452, !4, i64 3468, !4, i64 3724, !4, i64 7820, !4, i64 7884, !4, i64 25886, !4, i64 43888, !4, i64 45436, !4, i64 51628, !4, i64 57820, !4, i64 64012, !6, i64 64036, !6, i64 64040, !6, i64 64044, !6, i64 64048, !6, i64 64052, !6, i64 64056, !6, i64 64060, !6, i64 64064, !6, i64 64068, !6, i64 64072, !6, i64 64076, !6, i64 64080, !6, i64 64084, !6, i64 64088, !6, i64 64092, !6, i64 64096, !6, i64 64100, !6, i64 64104, !6, i64 64108, !6, i64 64112, !6, i64 64116, !3, i64 64120, !3, i64 64128, !3, i64 64136}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"int", !4, i64 0}
