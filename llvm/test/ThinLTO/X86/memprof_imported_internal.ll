;; Test to make sure that the memprof ThinLTO backend finds the correct summary
;; for an imported promoted local, so that we can perform the correct cloning.
;; In particular, we should be able to use the thinlto_src_file metadata to
;; recreate its original GUID. In particular, this test contains promoted
;; internal functions with the same original name as those that were imported,
;; and we want to ensure we don't use those by mistake.

;; The original code looks something like:
;;
;; src1.cc:
;; extern void external1();
;; extern void external2();
;; static void internal1() {
;;   external2();
;; }
;; static void internal2() {
;;   external2();
;; }
;; int main() {
;;   internal1();
;;   internal2();
;;   external1();
;;   return 0;
;; }
;;
;; src2.cc:
;; extern void external2();
;; static void internal1() {
;;   external2();
;; }
;; static void internal2() {
;;   external2();
;; }
;; void external1() {
;;   internal1();
;;   internal2();
;; }
;;
;; The assembly for src1 shown below was dumped after function importing, with
;; some hand modification to ensure we import the definitions of src2.cc's
;; external1 and internal1 functions, and the declaration only for its
;; internal2 function. I also hand modified it to add !callsite metadata
;; to a few calls, and the distributed ThinLTO summary in src1.o.thinlto.ll to
;; contain callsite metadata records with cloning results.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as src1.ll -o src1.o
; RUN: llvm-as src1.o.thinlto.ll -o src1.o.thinlto.bc

; RUN: opt -passes=memprof-context-disambiguation src1.o -S -memprof-import-summary=src1.o.thinlto.bc | FileCheck %s

;; Per the cloning results in the summary, none of the original functions should
;; call any memprof clones.
; CHECK-NOT: memprof
;; We should have one clone of src1.cc's internal1 that calls a clone of
;; external2.
; CHECK-LABEL: define void @_ZL9internal1v.llvm.5985484347676238233.memprof.1()
; CHECK:  tail call void @_Z9external2v.memprof.1()
; CHECK-LABEL: declare void @_Z9external2v.memprof.1()
;; We should have one clone of external1 that calls a clone of internal2 from
;; a synthesized callsite record (for a tail call with a missing frame).
; CHECK-LABEL: define available_externally {{.*}} void @_Z9external1v.memprof.1()
; CHECK:  tail call void @_ZL9internal1v.llvm.3267420853450984672()
; CHECK:  tail call void @_ZL9internal2v.llvm.3267420853450984672.memprof.1()
; CHECK-LABEL: declare void @_ZL9internal2v.llvm.3267420853450984672.memprof.1()
;; We should have one clone of src2.cc's internal1 function, calling a single
;; clone of external2, and a second clone that was detected to be a duplicate
;; of the first that becomes a declaration (since this is available_externally -
;; in the module with the prevailing copy it would be an alias to clone 1).
; CHECK-LABEL: define available_externally void @_ZL9internal1v.llvm.3267420853450984672.memprof.1()
; CHECK:  tail call void @_Z9external2v.memprof.1()
; CHECK:  tail call void @_Z9external2v.memprof.1()
; CHECK: declare void @_ZL9internal1v.llvm.3267420853450984672.memprof.2()
; CHECK-NOT: memprof

;--- src1.ll
; ModuleID = 'src1.o'
source_filename = "src1.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i32 @main() {
entry:
  tail call void @_ZL9internal1v.llvm.5985484347676238233()
  tail call void @_ZL9internal2v.llvm.5985484347676238233()
  tail call void @_Z9external1v()
  ret i32 0
}

define void @_ZL9internal1v.llvm.5985484347676238233() {
entry:
  tail call void @_Z9external2v(), !callsite !8
  ret void
}

define void @_ZL9internal2v.llvm.5985484347676238233() {
entry:
  tail call void @_Z9external2v()
  ret void
}

declare void @_Z9external2v()

define available_externally dso_local void @_Z9external1v() !thinlto_src_module !6 !thinlto_src_file !7 {
entry:
  tail call void @_ZL9internal1v.llvm.3267420853450984672()
  tail call void @_ZL9internal2v.llvm.3267420853450984672()
  ret void
}

define available_externally void @_ZL9internal1v.llvm.3267420853450984672() !thinlto_src_module !6 !thinlto_src_file !7 {
entry:
  ;; This one has more callsite records than the other version of internal1,
  ;; which would cause the code to iterate past the end of the callsite
  ;; records if we incorrectly got the other internal1's summary.
  tail call void @_Z9external2v(), !callsite !9
  tail call void @_Z9external2v(), !callsite !10
  ret void
}

declare void @_ZL9internal2v.llvm.3267420853450984672()

!6 = !{!"src2.o"}
!7 = !{!"src2.cc"}
!8 = !{i64 12345}
!9 = !{i64 23456}
!10 = !{i64 34567}

;--- src1.o.thinlto.ll
; ModuleID = 'src1.o.thinlto.bc'
source_filename = "src1.o.thinlto.bc"

^0 = module: (path: "src1.o", hash: (1393604173, 1072112025, 2857473630, 2016801496, 3238735916))
^1 = module: (path: "src2.o", hash: (760755700, 1705397472, 4198605753, 677969311, 2408738824))
;; src2.o:internal1. It specifies that we should have 3 clones total (including
;; original).
^3 = gv: (guid: 1143217136900127394, summaries: (function: (module: ^1, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^6, tail: 1), (callee: ^6, tail: 1)), callsites: ((callee: ^6, clones: (0, 1, 1), stackIds: (23456)), (callee: ^6, clones: (0, 1, 1), stackIds: (34567))))))
;; src2.o:internal2. It was manually modified to have importType = declaration.
^4 = gv: (guid: 3599593882704738259, summaries: (function: (module: ^1, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: declaration), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^6, tail: 1)))))
;; src1.o:internal1.
^5 = gv: (guid: 6084810090198994915, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^6, tail: 1)), callsites: ((callee: ^6, clones: (0, 1), stackIds: (12345))))))
^6 = gv: (guid: 8596367375252297795)
;; src1.o:internal2.
^7 = gv: (guid: 11092151021205906565, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^6, tail: 1)))))
;; src2.o:external1. It contains a synthesized callsite record for the tail call
;; to internal2 (the empty stackId list indicates it is synthesized for a
;; discovered missing tail call frame.
^8 = gv: (guid: 12313225385227428720, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, calls: ((callee: ^3, tail: 1), (callee: ^4, tail: 1)), callsites: ((callee: ^4, clones: (0, 1), stackIds: ())))))
;; src1.o:main.
^9 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 4, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^5, tail: 1), (callee: ^7, tail: 1), (callee: ^8, tail: 1)))))
^10 = flags: 97
^11 = blockcount: 0
