; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx72 \
; RUN:   | FileCheck %s

;; Regression test for the DwarfDebug::beginInstruction dedup gate plus the
;; recordTargetSameSourceLine hook.
;;
;; Two instructions share the same primary source coordinate (line 10, col 5,
;; same scope) but carry DIFFERENT irlayers. With the old code,
;; isSameSourceLocation compared getRawIRLayers(), so the dedup gate did NOT
;; fire for the second instruction: the primary .loc was re-emitted and the
;; second .loc_intermediate followed it.
;;
;; After this change isSameSourceLocation ignores irlayers. For the second
;; instruction the dedup gate fires (same primary position), so the primary
;; .loc is NOT re-emitted. The recordTargetSameSourceLine hook fires instead,
;; giving NVPTX a chance to emit .loc_intermediate even though no new line-
;; table row was needed. The third instruction (ret) shares the coordinate and
;; has no layers; the hook fires, the walk finds nothing, and nothing is emitted.
;;
;; Expected shape:
;;   .loc 1 10 5              <- first instruction (add), primary
;;   .loc_intermediate … 100  <- first instruction, layer
;;   [no second .loc 1 10]    <- dedup gate fires for mul
;;   .loc_intermediate … 200  <- second instruction, layer via hook

target triple = "nvptx64-nvidia-cuda"

define i32 @dedup_gate_demo(i32 %a, i32 %b) !dbg !5 {
  %1 = add i32 %a, %b, !dbg !20
  %2 = mul i32 %1, 3,  !dbg !21
  ret i32 %2,          !dbg !100
}

; CHECK-LABEL: dedup_gate_demo
;; First instruction (add): primary .loc plus its intermediate @ line 100.
; CHECK:      .loc 1 10 5
; CHECK-NEXT: .loc_intermediate {{[0-9]+}} 100 1
;; Second instruction (mul) shares the source coordinate, so the primary .loc is
;; NOT repeated -- but its layer differs, so it still gets its own
;; .loc_intermediate.
; CHECK-NOT:  .loc 1 10
; CHECK:      .loc_intermediate {{[0-9]+}} 200 1

!llvm.dbg.cu                    = !{!2}
!llvm.module.flags              = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3,
                              emissionKind: DebugDirectivesOnly)
!3 = !DIFile(filename: "demo.c", directory: "/tmp")
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "dedup_gate_demo", scope: !3, file: !3,
                            line: 1, type: !4, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !2)

!10 = !DIFile(filename: "demo.tile.ir", directory: "/tmp", checksumkind: CSK_MD5, checksum: "dddddddddddddddddddddddddddddddd", source: "demo tile ir text")

;; Two distinct intermediate layers on the shared source coordinate.
!110 = !DILayerLoc(line: 100, column: 1, file: !10, kind: "tile ir")
!111 = !DILayerLoc(line: 200, column: 1, file: !10, kind: "tile ir")
!120 = !DILayerLocList(!110)
!121 = !DILayerLocList(!111)

;; Same source (line 10, col 5, scope !5) but different layer lists.
!20 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !120)
!21 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !121)

;; Bare source location (no layers).
!100 = !DILocation(line: 10, column: 5, scope: !5)
