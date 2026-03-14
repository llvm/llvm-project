; REQUIRES: x86-registered-target
; RUN: %llc_dwarf -split-dwarf-file=%t.dwo < %s | FileCheck %s

; Ensure function-local DW_TAG_imported_declaration get skipped if its parent subprogram was not emitted.
; CHECK-NOT: DW_TAG_imported_declaration

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f1() !dbg !13 {
lbl:
  ret void, !dbg !16
}

define void @f2() !dbg !22 {
lbl:
  ret void, !dbg !23
}

!llvm.dbg.cu = !{!0, !2, !10}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cc", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "b.cc", directory: "")
!4 = !{!5}
!5 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !6, entity: !7)
!6 = !DISubprogram(scope: null, spFlags: DISPFlagOptimized, retainedNodes: !4)
!7 = !DINamespace(scope: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11, emissionKind: FullDebug)
!11 = !DIFile(filename: "c.cc", directory: "")
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(scope: null, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DILocation(line: 0, scope: !17, inlinedAt: !18)
!17 = distinct !DISubprogram(scope: null, unit: !10)
!18 = !DILocation(line: 0, scope: !21)
!21 = !DILexicalBlockFile(scope: !13, discriminator: 0)
!22 = distinct !DISubprogram(scope: null, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!23 = !DILocation(line: 0, scope: !24, inlinedAt: !25)
!24 = distinct !DISubprogram(scope: null, unit: !2)
!25 = !DILocation(line: 0, scope: !22)
