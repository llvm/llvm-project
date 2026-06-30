; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx72 \
; RUN:   | FileCheck %s

;; Regression test for the DwarfDebug::beginInstruction dedup gate.
;;
;; DebugLoc::isSameSourceLocation early-exited true when two DebugLocs
;; shared the same primary DILocation pointer, regardless of differences
;; in their secondary intermediate-location operand. As a result, two
;; consecutive MachineInstrs whose !dbg attachments wrapped the SAME
;; primary in different MDTuples (different intermediates) would be
;; deduped, and the second .loc_intermediate would be silently dropped
;; from the emitted PTX.
;;
;; This test pins both .loc_intermediate directives into the output so
;; the dedup gate is forced to consider intermediate-loc differences in
;; its equivalence check.

target triple = "nvptx64-nvidia-cuda"

define i32 @dedup_gate_demo(i32 %a, i32 %b) !dbg !5 {
  %1 = add i32 %a, %b, !dbg !20
  %2 = mul i32 %1, 3,  !dbg !21
  ret i32 %2,          !dbg !100
}

;; Both instructions share the primary DILocation (line 10, col 5), so
;; only one .loc 1 10 5 should appear. But the two distinct intermediate
;; DILocations (line 100 and line 200) MUST each be emitted as their own
;; .loc_intermediate, in source order.

; CHECK-LABEL: dedup_gate_demo
;; First instruction (add): primary + intermediate @ line 100.
; CHECK:      .loc 1 10 5
; CHECK-NEXT: .loc_intermediate {{[0-9]+}} 100 1
;; Second instruction (mul): primary must be re-emitted (proves the dedup
;; gate did NOT fire, because the intermediate operand differs) and the
;; intermediate @ line 200 must follow.
; CHECK:      .loc 1 10 5
; CHECK-NEXT: .loc_intermediate {{[0-9]+}} 200 1

!llvm.dbg.cu                    = !{!2}
!llvm.module.flags              = !{!0, !1}
!llvm.intermediate_level_source = !{!200}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3,
                              emissionKind: DebugDirectivesOnly)
!3 = !DIFile(filename: "demo.c", directory: "/tmp")
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "dedup_gate_demo", scope: !3, file: !3,
                            line: 1, type: !4, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !2)

!10 = !DIFile(filename: "demo.tile.ir", directory: "/tmp")
!12 = distinct !DISubprogram(name: "dedup_gate_demo_tile", scope: !10,
                             file: !10, line: 1, type: !4, scopeLine: 1,
                             spFlags: DISPFlagDefinition, unit: !2)

;; Primary DILocation — shared between !20 and !21 (uniqued by content).
!100 = !DILocation(line: 10, column: 5, scope: !5)

;; Two distinct intermediate DILocations.
!110 = !DILocation(line: 100, column: 1, scope: !12)
!111 = !DILocation(line: 200, column: 1, scope: !12)

;; MDTuple #1: primary !100 + intermediate "tile ir" @ line 100.
!20 = !{!100, !{!"tile ir", !110}}
;; MDTuple #2: SAME primary !100 + intermediate "tile ir" @ line 200.
!21 = !{!100, !{!"tile ir", !111}}

!200 = !{!"tile ir", !10,
         !"// tile-IR line 100\n// tile-IR line 200\n"}
