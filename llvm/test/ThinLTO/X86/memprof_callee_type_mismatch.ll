;; Test to ensure the callite when updated to call a clone does not mutate the
;; callee function type. In rare cases we may end up with a callee declaration
;; that does not match the call type, because it was imported from a different
;; module with an incomplete return type (in which case clang gives it a void
;; return type).

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as src.ll -o src.o
; RUN: llvm-as src.o.thinlto.ll -o src.o.thinlto.bc
; RUN: opt -passes=memprof-context-disambiguation src.o -S -memprof-import-summary=src.o.thinlto.bc | FileCheck %s

;--- src.ll
; ModuleID = 'src.o'
source_filename = "src.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(ptr %b) {
entry:
  ;; This call is not changed as the summary specifies clone 0.
  ; CHECK: call ptr @_Z3foov()
  %call = call ptr @_Z3foov(), !callsite !5
  ;; After changing this call to call a clone, the function type should still
  ;; be ptr, despite the void on the callee declaration.
  ; CHECK: call ptr @_Z3foov.memprof.1()
  %call1 = call ptr @_Z3foov(), !callsite !6
  %0 = load ptr, ptr %b, align 8
  ;; Although the summary indicates this should call clone 1, and the VP
  ;; metadata indicates the callee is _Z3foov, it is not updated because
  ;; the ICP facility requires the function types to match.
  ; CHECK: call ptr %0()
  %call2 = call ptr %0(), !prof !7, !callsite !8
  ret i32 0
}

;; Both the original callee function declaration and its clone have void return
;; type.
; CHECK: declare void @_Z3foov()
; CHECK: declare void @_Z3foov.memprof.1()
declare void @_Z3foov()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git (git@github.com:llvm/llvm-project.git e391301e0e4d9183fe06e69602e87b0bc889aeda)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src.cc", directory: "", checksumkind: CSK_MD5, checksum: "8636c46e81402013b9d54e8307d2f149")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!5 = !{i64 8632435727821051414}
!6 = !{i64 -3421689549917153178}
!7 = !{!"VP", i32 0, i64 4, i64 9191153033785521275, i64 4}
!8 = !{i64 1234}

;--- src.o.thinlto.ll
; ModuleID = 'src.o.thinlto.bc'
source_filename = "src.o.thinlto.bc"

^0 = module: (path: "src.o", hash: (2823430083, 3994560862, 899296057, 1055405378, 2961356784))
^1 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), callsites: ((callee: null, clones: (0), stackIds: (8632435727821051414)), (callee: null, clones: (1), stackIds: (15025054523792398438)), (callee: null, clones: (1), stackIds: (1234))))))
^2 = flags: 353
^3 = blockcount: 0
