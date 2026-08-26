; RUN: llvm-as -disable-output %s -o - 2>&1 | FileCheck --allow-empty %s
;;
;; Verification of DILayerLoc is structural only, so a checksum-less intermediate
;; file verifies clean -- with or without source.

; CHECK-NOT: requires a checksum
; CHECK-NOT: ignoring invalid debug info

define void @k(ptr %p) !dbg !5 {
  store ptr null, ptr %p, align 8, !dbg !20
  store ptr null, ptr %p, align 8, !dbg !21
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)

;; No checksum, no source.
!10 = !DIFile(filename: "kernel.tileir", directory: ".")
!11 = !DILayerLoc(line: 42, column: 5, file: !10, kind: "tile ir")
!12 = !DILayerLocList(!11)

;; Source but still no checksum -- the case NVPTX would otherwise use a digest
;; for, and the one that used to be rejected outright.
!13 = !DIFile(filename: "kernel.gpuir", directory: ".", source: "gpu ir source")
!14 = !DILayerLoc(line: 7, column: 1, file: !13, kind: "gpu ir")
!15 = !DILayerLocList(!14)

!20 = !DILocation(line: 2, column: 5, scope: !5, irlayers: !12)
!21 = !DILocation(line: 3, column: 5, scope: !5, irlayers: !15)
