;; Test that quoted string fields in ThinLTO module and combined summaries
;; properly escape special characters (such as \01, quotes, and backslashes)
;; when disassembled and round-trip through llvm-as and llvm-dis.

; RUN: split-file %s %t
; RUN: opt %t/module.ll -module-summary -o - | llvm-dis -o - | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=MODULE
; RUN: llvm-as %t/combined.ll -o - | llvm-dis -o - | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=COMBINED

; MODULE-DAG: = gv: (name: "\01fn\22\\1", summaries: (function:
; MODULE-DAG: = gv: (name: "\01vt\22\\1", summaries: (variable:
; MODULE-DAG: = typeidCompatibleVTable: (name: "\01typeid_vt\22\\1", summary:

; COMBINED: ^0 = module: (path: "\01mod\22\\1.o", hash: (0, 0, 0, 0, 0))
; COMBINED: ^1 = gv: (guid: 100, summaries: (function: (module: ^0, flags: ({{.*}}), insts: 1, typeIdInfo: (typeTests: (^3)))))
; COMBINED: ^2 = gv: (guid: 200, summaries: (function: (module: ^0, flags: ({{.*}}), insts: 1)))
; COMBINED: ^3 = typeid: (name: "\01tid\22\\1", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7), wpdResolutions: ((offset: 0, wpdRes: (kind: singleImpl, singleImplName: "\01impl\22\\1"))))) ; guid =
; COMBINED: ^4 = cfiFunctionDefs: ((guid: 100, name: "\01def\22\\1"))
; COMBINED: ^5 = cfiFunctionDecls: ((guid: 200, name: "\01decl\22\\1"))
; COMBINED: ^6 = blockcount: 0

;--- module.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@"\01vt\22\\1" = constant [1 x ptr] [ptr @"\01fn\22\\1"], !type !0

define void @"\01fn\22\\1"() {
  ret void
}

!0 = !{i64 0, !"\01typeid_vt\22\\1"}

;--- combined.ll
^0 = module: (path: "\01mod\22\\1.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 100, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1, typeIdInfo: (typeTests: (^3)))))
^2 = gv: (guid: 200, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
^3 = typeid: (name: "\01tid\22\\1", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7), wpdResolutions: ((offset: 0, wpdRes: (kind: singleImpl, singleImplName: "\01impl\22\\1")))))
^4 = cfiFunctionDefs: ((guid: 100, name: "\01def\22\\1"))
^5 = cfiFunctionDecls: ((guid: 200, name: "\01decl\22\\1"))
^6 = blockcount: 0
