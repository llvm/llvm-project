;; Test that call graph edges from indirect calls (from value profile metadata
;; or !callees metadata) are marked as indirect in the per-module function
;; summary, and that the flag round trips through bitcode and the textual
;; summary. The combined index encoding is tested in
;; test/Assembler/thinlto-summary.ll.

; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s --check-prefix=BC
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
;; Round trip the textual summary through the assembler and disassembler.
; RUN: llvm-dis -o - %t.o | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; The call edge flags are encoded as:
;;   hotness (3 bits) | tailcall << 3 | indirectcall << 4
;; There is no profile summary, so the hotness is none (2) for the profiled
;; indirect call targets and unknown (0) for the other calls. The edge to
;; direct1, which is called both directly and indirectly, is marked indirect,
;; and has the max hotness.
; BC: <PERMODULE_PROFILE {{.*}} op7={{[0-9]+}} op8=18 op9={{[0-9]+}} op10=18 op11={{[0-9]+}} op12=16/>
; BC-NEXT: <PERMODULE_PROFILE {{.*}} op7={{[0-9]+}} op8=0/>

; DIS-DAG: ^[[DIRECT1:[0-9]+]] = gv: (name: "direct1")
; DIS-DAG: ^[[INDIRECT2:[0-9]+]] = gv: (name: "indirect2")
; DIS-DAG: ^[[CALLEE:[0-9]+]] = gv: (name: "callee")
; DIS-DAG: ^[[DIRECT2:[0-9]+]] = gv: (name: "direct2")
; DIS-DAG: gv: (name: "caller", {{.*}} calls: ((callee: ^[[DIRECT1]], hotness: none, indirect: 1), (callee: ^[[INDIRECT2]], hotness: none, indirect: 1), (callee: ^[[CALLEE]], indirect: 1))
; DIS-DAG: gv: (name: "directcaller", {{.*}} calls: ((callee: ^[[DIRECT2]])){{[,)]}}

define void @caller(ptr %fp, ptr %fp2) {
entry:
  call void @direct1()
  call void %fp(), !prof !0
  call void %fp2(), !callees !1
  ret void
}

define void @directcaller() {
entry:
  call void @direct2()
  ret void
}

declare void @direct1()
declare void @direct2()
declare void @indirect2()
declare void @callee()

;; Two profiled targets, direct1 and indirect2.
!0 = !{!"VP", i32 0, i64 10, i64 6701688212809690871, i64 6, i64 -8888628552600679731, i64 4}
!1 = !{ptr @callee}
