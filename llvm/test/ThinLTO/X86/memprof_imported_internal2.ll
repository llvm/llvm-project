;; Test to make sure that the memprof ThinLTO backend finds the correct summary
;; when there was a naming conflict between an internal function in the original
;; module and an imported external function with the same name. The IR linking
;; will automatically append a "." followed by a numbered suffix to the existing
;; local name in that case. Note this can happen with C where the mangling would
;; be the same for the internal and external functions of the same name (C++
;; would have different mangling).

;; Note we don't need any MemProf related metadata for this to fail to find a
;; ValueInfo and crash if the wrong GUID is computed for the renamed local.

;; The original code looks something like:
;;
;; src1.c:
;; extern void external1();
;; extern void external2();
;; static void foo() {
;;   external2();
;; }
;; int main() {
;;   external1();
;;   foo();
;;   return 0;
;; }
;;
;; src2.c:
;; extern void external2();
;; void foo() {
;;   external2();
;; }
;; void external1() {
;;   foo();
;; }
;;
;; The assembly for src1 shown below was dumped after function importing.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as src1.ll -o src1.o
; RUN: llvm-as src1.o.thinlto.ll -o src1.o.thinlto.bc

;; Simply check that we don't crash when trying to find the ValueInfo for each
;; function in the IR.
; RUN: opt -passes=memprof-context-disambiguation src1.o -S -memprof-import-summary=src1.o.thinlto.bc

;--- src1.ll
; ModuleID = 'src1.o'
source_filename = "src1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i32 @main() {
entry:
  tail call void @external1()
  tail call void @foo.2()
  ret i32 0
}

define internal void @foo.2() {
entry:
  tail call void @external2()
  ret void
}

declare void @external2()

define available_externally dso_local void @foo() !thinlto_src_module !1 !thinlto_src_file !2 {
entry:
  tail call void @external2()
  ret void
}

define available_externally dso_local void @external1() !thinlto_src_module !1 !thinlto_src_file !2 {
entry:
  tail call void @foo()
  ret void
}

!1 = !{!"src2.o"}
!2 = !{!"src2.c"}

;--- src1.o.thinlto.ll
; ModuleID = 'src1.o.thinlto.bc'
source_filename = "src1.o.thinlto.bc"

^0 = module: (path: "src1.o", hash: (2435941910, 498944982, 2551913764, 2759430100, 3918124321))
^1 = module: (path: "src2.o", hash: (1826286437, 1557684621, 1220464477, 2734102338, 1025249503))
^2 = module: (path: "src3.o", hash: (1085916433, 503665945, 2163560042, 340524, 2255774964))
^3 = gv: (guid: 1456206394295721279, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0)))) ;; src1.c:foo
^4 = gv: (guid: 6699318081062747564, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0)))) ;; src2.c:foo
^5 = gv: (guid: 13087145834073153720, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^4, tail: 1))))) ;; src1.c:external1
^6 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^5, tail: 1), (callee: ^3, tail: 1))))) ;; src1.c:main
^8 = flags: 97
^9 = blockcount: 0
