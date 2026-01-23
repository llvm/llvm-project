;; Check that the values of -icp-max-num-vtables, -icp-max-prom, and
;; -module-summary-max-indirect-edges affect the number of profiled
;; vtables and virtual functions propagated from the VP metadata to
;; the ThinLTO summary as expected.

;; First try with a max of 1 for both vtables and virtual functions.
; RUN: opt -module-summary -icp-max-num-vtables=1 -icp-max-prom=1 %s -o %t.o

; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s --check-prefix=SUMMARY

; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
;; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.o | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

;; Next check that a larger -module-summary-max-indirect-edges value overrides
;; -icp-max-prom when determining how many virtual functions to summarize.
; RUN: opt -module-summary -icp-max-num-vtables=1 -icp-max-prom=1 -module-summary-max-indirect-edges=2 %s -o %t2.o
; RUN: llvm-bcanalyzer -dump %t2.o | FileCheck %s --check-prefixes=SUMMARY,SUMMARY2
; RUN: llvm-dis -o - %t2.o | FileCheck %s --check-prefixes=DIS,DIS2

; SUMMARY: 	  <GLOBALVAL_SUMMARY_BLOCK
; SUMMARY-NEXT:   <VERSION op0=
; SUMMARY-NEXT:   <FLAGS op0=0/>

;; The `VALUE_GUID` below represents the "_ZTV4Base" referenced by the instruction
;; that loads vtable pointers.
; SUMMARY-NEXT:   <VALUE_GUID {{.*}} op0=[[VTABLEBASE:[0-9]+]] op1=456547254 op2=3929380924/>
;; The `VALUE_GUID` below represents the "_ZN4Base4funcEv" referenced by the
;; indirect call instruction.
; SUMMARY-NEXT:   <VALUE_GUID {{.*}} op0=[[VFUNCBASE:[0-9]+]] op1=1271117309 op2=2009351347/>
;; The `VALUE_GUID` below represents the "_ZN7Derived4funcEv" referenced by the
;; indirect call instruction.
; SUMMARY2-NEXT:  <VALUE_GUID {{.*}} op0=[[VFUNCDER:[0-9]+]] op1=1437699922 op2=4037658799/>

;; <PERMODULE_PROFILE> has the format [valueid, flags, instcount, funcflags,
;;                                     numrefs, rorefcnt, worefcnt,
;;                                     m x valueid,
;;                                     n x (valueid, hotness+tailcall)]
;; NOTE vtables and functions from Derived class are dropped in the base case
;; because `-icp-max-num-vtables` and `-icp-max-prom` are both set to one.
; SUMMARY-NEXT:   <PERMODULE_PROFILE {{.*}} op0=0 op1=0 op2=4 op3=256 op4=1 op5=1 op6=0 op7=[[VTABLEBASE]] op8=[[VFUNCBASE]] op9=3
;; With -module-summary-max-indirect-edges=2 we do get the Derived class
;; function in the summary.
; SUMMARY2-SAME:  op10=[[VFUNCDER]] op11=2
;; We should have no other ops before the end of the summary record.
; SUMMARY-NOT:	  op
; SUMMARY-SAME:	  />
; SUMMARY-NEXT:   </GLOBALVAL_SUMMARY_BLOCK>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Function has one BB and an entry count of 150, so the BB is hot according to
;; ProfileSummary and reflected so in the bitcode (see llvm-dis output).
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
;; 1960855528937986108 is the MD5 hash of _ZTV4Base, and
;; 13870436605473471591 is the MD5 hash of _ZTV7Derived
!16 = !{!"VP", i32 2, i64 150, i64 1960855528937986108, i64 100, i64 13870436605473471591, i64 50}
;; 5459407273543877811 is the MD5 hash of _ZN4Base4funcEv, and
;; 6174874150489409711 is the MD5 hash of  _ZN7Derived4funcEv
!17 = !{!"VP", i32 0, i64 150, i64 5459407273543877811, i64 100, i64 6174874150489409711, i64 50}

;; ModuleSummaryIndex stores <guid, global-value summary> map in std::map; so
;; global value summaries are printed out in the order that gv's guid increases.
; DIS:	^[[VTABLEBASE2:[0-9]+]] = gv: (guid: 1960855528937986108)
; DIS:	^[[VFUNCBASE2:[0-9]+]] = gv: (guid: 5459407273543877811)
; DIS2:	^[[VFUNCDER2:[0-9]+]] = gv: (guid: 6174874150489409711)
; DIS:	gv: (name: "_Z4testP4Base", {{.*}} calls: ((callee: ^[[VFUNCBASE2]], hotness: hot)
;; With -module-summary-max-indirect-edges=2 we get the Derived func.
; DIS2-SAME:	(callee: ^[[VFUNCDER2]], hotness: none)
; DIS-NOT:	callee
; DIS-SAME:	), refs: (readonly ^[[VTABLEBASE2]])
