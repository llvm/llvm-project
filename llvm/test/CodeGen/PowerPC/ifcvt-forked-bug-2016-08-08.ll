; ModuleID = 'bugpoint-reduced-instructions.bc'
; RUN: llc -O2 -o - %s -verify-machineinstrs | FileCheck %s
source_filename = "bugpoint-output-9ad75f8.bc"
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define hidden void @_ZN11__sanitizer25MaybeStartBackgroudThreadEv() local_unnamed_addr #0 {
entry:
  br i1 undef, label %land.lhs.true, label %if.end

; CHECK: # %land.lhs.true
; Test updated due D63152 where any load/store prevents shrink-wrapping
; CHECK-NEXT: bc
; CHECK-NEXT: # %if.end4
land.lhs.true:                                    ; preds = %entry
  br i1 undef, label %return, label %if.end4

if.end:                                           ; preds = %entry
  br i1 icmp ne (i32 (i8*, i8*, i8* (i8*)*, i8*)* @_ZN11__sanitizer19real_pthread_createEPvS0_PFS0_S0_ES0_, i32 (i8*, i8*, i8* (i8*)*, i8*)* null), label %if.end4, label %return

if.end4:                                          ; preds = %if.end, %land.lhs.true
  %call5 = tail call i8* @_ZN11__sanitizer21internal_start_threadEPFvPvES0_(void (i8*)* nonnull @_ZN11__sanitizer16BackgroundThreadEPv, i8* null) #7
  unreachable

return:                                           ; preds = %if.end, %land.lhs.true
  ret void
}

declare extern_weak signext i32 @_ZN11__sanitizer19real_pthread_createEPvS0_PFS0_S0_ES0_(i8*, i8*, i8* (i8*)*, i8*) #2

declare i8* @_ZN11__sanitizer21internal_start_threadEPFvPvES0_(void (i8*)*, i8*) local_unnamed_addr #2

declare hidden void @_ZN11__sanitizer16BackgroundThreadEPv(i8* nocapture readnone) #5

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nobuiltin nounwind }
