; RUN: opt %loadPolly -polly-print-ast -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
;
; Verify we do not simplify the runtime check to "true" due to the domain
; constraints as the test contains an error block that influenced the domains
; already.
;
; CHECK: if (p_0_loaded_from_this <= -1 || p_0_loaded_from_this >= 1)
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque

$_ZNKSt5ctypeIcE5widenEc = comdat any

; Function Attrs: uwtable
define weak_odr signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"* %this, i8 signext %__c) #0 comdat align 2 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %_M_widen_ok = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %this, i64 0, i32 8
  %0 = load i8, i8* %_M_widen_ok, align 8, !tbaa !1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry.split
  %idxprom = zext i8 %__c to i64
  %arrayidx = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %this, i64 0, i32 9, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1, !tbaa !7
  br label %return

if.end:                                           ; preds = %entry.split
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* %this)
  %2 = bitcast %"class.std::ctype"* %this to i8 (%"class.std::ctype"*, i8)***
  %vtable = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %2, align 8, !tbaa !8
  %vfn = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable, i64 6
  %3 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn, align 8
  %call = tail call signext i8 %3(%"class.std::ctype"* %this, i8 signext %__c)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i8 [ %1, %if.then ], [ %call, %if.end ]
  ret i8 %retval.0
}

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0"}
!1 = !{!2, !4, i64 56}
!2 = !{!"_ZTSSt5ctypeIcE", !3, i64 16, !6, i64 24, !3, i64 32, !3, i64 40, !3, i64 48, !4, i64 56, !4, i64 57, !4, i64 313, !4, i64 569}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"bool", !4, i64 0}
!7 = !{!4, !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"vtable pointer", !5, i64 0}
