; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s
;;
;; Verifier rule: an instruction with a tuple-form !dbg chain (i.e. a
;; TileIR-style intermediate-loc) requires the module to also declare the
;; corresponding source content via !llvm.intermediate_level_source.
;; The diagnostic must be emitted exactly once per module, even though many
;; instructions may carry the chain (latched in Verifier::verifyIntermediateLocChain).

; CHECK:     intermediate-loc !dbg chain found but module has no !llvm.intermediate_level_source named metadata
; CHECK-NOT: intermediate-loc !dbg chain found but module has no !llvm.intermediate_level_source named metadata
; CHECK:     warning: ignoring invalid debug info

define void @two_chains() !dbg !6 {
  ret void, !dbg !20
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3, !4}
;; Note: NO !llvm.intermediate_level_source — that is the exact precondition
;; that should cause the verifier to fail.

!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !5, producer: "p", emissionKind: LineTablesOnly)
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIFile(filename: "src.c", directory: ".")
!6 = distinct !DISubprogram(name: "two_chains", scope: !5, file: !5, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}

;; Two intermediate-loc chains. Without latching, each would produce a
;; redundant "missing named metadata" diagnostic; with latching only the
;; first fires.
!20 = !{!11, !12}
!11 = !DILocation(line: 1, scope: !6)
!12 = !{!"tile ir", !13}
!13 = !DILocation(line: 1, scope: !14)
!14 = distinct !DILexicalBlockFile(scope: !6, file: !15, discriminator: 0)
!15 = !DIFile(filename: "ti", directory: ".")

!21 = !{!16, !12}
!16 = !DILocation(line: 2, scope: !6)
