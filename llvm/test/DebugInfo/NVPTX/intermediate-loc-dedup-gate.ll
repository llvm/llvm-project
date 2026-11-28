; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx72 \
; RUN:   | FileCheck %s

;; Two consecutive instructions at the same source position but with different
;; layer lists. The primary .loc is not repeated, since the line table needs no
;; new row, but the second instruction still gets its own .loc_intermediate.

target triple = "nvptx64-nvidia-cuda"

define i32 @dedup_gate_demo(i32 %a, i32 %b) !dbg !5 {
  %1 = add i32 %a, %b, !dbg !20
  %2 = mul i32 %1, 3,  !dbg !21
  ret i32 %2,          !dbg !100
}

; CHECK-LABEL: dedup_gate_demo
;; add: the primary .loc and its layer.
; CHECK:      .loc 1 10 5
; CHECK-NEXT: .loc_intermediate {{[0-9]+}} 100 1
;; mul: no second primary .loc, but its own layer.
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

!10 = !DIFile(filename: "demo.ir", directory: "/tmp", checksumkind: CSK_MD5, checksum: "dddddddddddddddddddddddddddddddd", source: "demo ir text")

;; Two layers for the one source position.
!110 = !DILayerLoc(line: 100, column: 1, file: !10, kind: "intermediate ir")
!111 = !DILayerLoc(line: 200, column: 1, file: !10, kind: "intermediate ir")
!120 = !DILayerLocList(!110)
!121 = !DILayerLocList(!111)

;; Same source position, different layer lists.
!20 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !120)
!21 = !DILocation(line: 10, column: 5, scope: !5, irlayers: !121)

;; The same source position without layers.
!100 = !DILocation(line: 10, column: 5, scope: !5)
