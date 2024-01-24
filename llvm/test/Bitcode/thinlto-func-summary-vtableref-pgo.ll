; RUN: opt -module-summary %s -o %t.o

; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS


; CHECK: <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:   <VERSION op0=9/>
; CHECK-NEXT:   <FLAGS op0=0/>
; The `VALUE_GUID` below represents the "_ZN4Base4funcEv" referenced by the
; indirect call instruction.
; CHECK-NEXT:   <VALUE_GUID op0=8 op1=5459407273543877811/>
; <PERMODULE_PROFILE> has the format [valueid, flags, instcount, funcflags,
;                                     numrefs, rorefcnt, worefcnt,
;                                     n x (valueid, hotness+tailcall)]
; CHECK-NEXT:   <PERMODULE_PROFILE abbrevid=4 op0=0 op1=0 op2=4 op3=256 op4=0 op5=0 op6=0 op7=8 op8=2/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z4testP4Base(ptr %0) !prof !0 {
  %2 = load ptr, ptr %0, !prof !1
  %3 = load ptr, ptr %2
  %4 = tail call i32 %3(ptr %0), !prof !2
  ret i32 %4
}

!0 = !{!"function_entry_count", i32 150}
; 1960855528937986108 is the MD5 hash of _ZTV4Base
!1 = !{!"VP", i32 2, i64 1600, i64 1960855528937986108, i64 1600}
; 5459407273543877811 is the MD5 hash of _ZN4Base4funcEv
!2 = !{!"VP", i32 0, i64 1600, i64 5459407273543877811, i64 1600}

; ModuleSummaryIndex stores <guid, global-value summary> map in std::map; so
; global value summares are printed out in the order that gv's guid increases.
; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (guid: 5459407273543877811)
; DIS: ^2 = gv: (name: "_Z4testP4Base", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 4, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 1, mustBeUnreachable: 0), calls: ((callee: ^1, hotness: none))))) ; guid = 15857150948103218965
; DIS: ^3 = blockcount: 0
