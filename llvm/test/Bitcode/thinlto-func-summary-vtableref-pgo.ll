; RUN: opt -module-summary %s -o %t.o

; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS


; CHECK: <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:   <VERSION op0=9/>
; CHECK-NEXT:   <FLAGS op0=0/>
; The `VALUE_GUID` below represents the "_ZN4Base4funcEv" referenced by the
; indirect call instruction.
; CHECK-NEXT:   <VALUE_GUID op0=17 op1=5459407273543877811/>
; <PERMODULE_PROFILE> has the format [valueid, flags, instcount, funcflags,
;                                     numrefs, rorefcnt, worefcnt,
;                                     n x (valueid, hotness+tailcall)]
; CHECK-NEXT:   <PERMODULE_PROFILE abbrevid=4 op0=0 op1=0 op2=4 op3=256 op4=0 op5=0 op6=0 op7=17 op8=3/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function has one BB and an entry count of 150, so the BB is hot according to
; ProfileSummary and reflected so in the bitcode (see llvm-dis output).
define i32 @_Z4testP4Base(ptr %0) !prof !15 {
  %2 = load ptr, ptr %0, !prof !16
  %3 = load ptr, ptr %2
  %4 = tail call i32 %3(ptr %0), !prof !17
  ret i32 %4
}

!llvm.module.flags = !{!1}


!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 200}
!6 = !{!"MaxInternalCount", i64 200}
!7 = !{!"MaxFunctionCount", i64 200}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 990000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}

!15 = !{!"function_entry_count", i32 150}
; 1960855528937986108 is the MD5 hash of _ZTV4Base
!16 = !{!"VP", i32 2, i64 1600, i64 1960855528937986108, i64 1600}
; 5459407273543877811 is the MD5 hash of _ZN4Base4funcEv
!17 = !{!"VP", i32 0, i64 1600, i64 5459407273543877811, i64 1600}

; ModuleSummaryIndex stores <guid, global-value summary> map in std::map; so
; global value summares are printed out in the order that gv's guid increases.
; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (guid: 5459407273543877811)
; DIS: ^2 = gv: (name: "_Z4testP4Base", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 4, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 1, mustBeUnreachable: 0), calls: ((callee: ^1, hotness: hot))))) ; guid = 15857150948103218965
; DIS: ^3 = blockcount: 0
