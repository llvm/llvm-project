;; This test checks whether DWARF tag DW_TAG_immutable_type
;; is accepted and processed.
; REQUIRES: default_target
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; Test whether DW_TAG_immutable_type is accepted.

; CHECK: DW_TAG_immutable_type

; ModuleID = 'immutable.d'
source_filename = "immutable.d"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@_D9immutable1aya = constant i8 97, align 1, !dbg !0 ; [#uses = 0]

!llvm.module.flags = !{!5}
!llvm.dbg.cu = !{!6}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", linkageName: "_D9immutable1aya", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !3, name: "immutable")
!3 = !DIFile(filename: "immutable.d", directory: "/home/luis/Temp")
!4 = !DIDerivedType(tag: DW_TAG_immutable_type, baseType: !14)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DICompileUnit(language: DW_LANG_D, file: !3, producer: "LDC 1.28.0 (LLVM 13.0.0)", isOptimized: false, runtimeVersion: 1, emissionKind: FullDebug, enums: !7, globals: !8, imports: !9)
!7 = !{}
!8 = !{!0}
!9 = !{!10}
!10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !11, file: !3)
!11 = !DIModule(scope: !12, name: "object")
!12 = !DIFile(filename: "usr/include/dlang/ldc/object.d", directory: "/")
!13 = !{!"ldc version 1.28.0"}
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_UTF)
