; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s
;;
;; Verifier rule: every intermediate DILocation referenced from a tuple-form
;; !dbg attachment must name a file declared in slot 1 of some entry of
;; !llvm.intermediate_level_source. Mismatched filenames otherwise cause
;; NVPTXDwarfDebug::buildIntermediateSourceSection to silently drop the
;; corresponding code_block (FileNum lookup misses), producing PTX with no
;; usable intermediate-source mapping (nvbug 6133285). The diagnostic is a
;; DI consistency error, so llvm-as warns and continues (matching the
;; convention of the existing tuple-!dbg shape check); the value of the rule
;; is surfacing the problem at IR load instead of in codegen.

; CHECK: intermediate DILocation references a file with no matching entry in !llvm.intermediate_level_source
; CHECK-NEXT: ret void, !dbg
;; The well-formed @good_match below must not produce another diagnostic.
; CHECK-NOT: intermediate DILocation references a file
; CHECK: warning: ignoring invalid debug info

define void @bad_mismatch() !dbg !6 {
  ret void, !dbg !9
}

define void @good_match() !dbg !26 {
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3, !4}
!llvm.intermediate_level_source = !{!0}

;; The named-metadata entry declares the intermediate file "name1".
!0 = !{!"tile ir", !25, !"src1"}

!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !5, producer: "p", emissionKind: LineTablesOnly)
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIFile(filename: "src.c", directory: ".")
!6 = distinct !DISubprogram(name: "bad_mismatch", scope: !5, file: !5, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!26 = distinct !DISubprogram(name: "good_match", scope: !5, file: !5, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}

;; --- Bad: intermediate DIFile is "ti", which is NOT in !llvm.intermediate_level_source.
!9  = !{!10, !11}
!10 = !DILocation(line: 1, scope: !6)
!11 = !{!"tile ir", !12}
!12 = !DILocation(line: 1, scope: !13)
!13 = distinct !DILexicalBlockFile(scope: !6, file: !14, discriminator: 0)
!14 = !DIFile(filename: "ti", directory: ".")

;; --- Good: intermediate DIFile is "name1", matching slot 1 of !0.
!20 = !{!21, !22}
!21 = !DILocation(line: 1, scope: !26)
!22 = !{!"tile ir", !23}
!23 = !DILocation(line: 1, scope: !24)
!24 = distinct !DILexicalBlockFile(scope: !26, file: !25, discriminator: 0)
!25 = !DIFile(filename: "name1", directory: ".")
