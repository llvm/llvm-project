; RUN: llvm-as < %s | llvm-dis | FileCheck %s
;;
;; Checks that irlayers survive a bitcode round-trip on a DILocation written as
;; a METADATA_LOCATION record -- one reachable only as an inlinedAt target, with
;; a layer list built through forward references.

define dso_local void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  store ptr %v, ptr %v, align 8, !dbg !20
  store ptr %v, ptr %v, align 8, !dbg !23
  ret void, !dbg !21
}

attributes #0 = { noinline optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 20, type: !9, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0)

!14 = !DIFile(filename: "kernel.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "tile ir text")
!15 = !DIFile(filename: "kernel.gpuir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "gpu ir text")

;; !18 precedes its entries, so they parse as forward references and the list is
;; re-uniqued once they resolve. !22 names the same entries after they resolve,
;; so it must land on !18's node -- a stale cached hash would miss the uniquing
;; lookup and produce a second, identical list.
!18 = !DILayerLocList(!16, !17)
!16 = !DILayerLoc(line: 100, column: 1, file: !14, kind: "tile ir")
!17 = !DILayerLoc(line: 7, column: 3, file: !15, kind: "gpu ir")
!22 = !DILayerLocList(!16, !17)

!19 = distinct !DILocation(line: 30, column: 1, scope: !8, irlayers: !18)
!24 = distinct !DILocation(line: 40, column: 1, scope: !8, irlayers: !22)
!20 = !DILocation(line: 21, column: 5, scope: !11, inlinedAt: !19)
!21 = !DILocation(line: 22, column: 1, scope: !11, inlinedAt: !19)
!23 = !DILocation(line: 23, column: 5, scope: !11, inlinedAt: !24)

;; The metadata-block location keeps its layers, and the two entries keep their
;; order.
; CHECK-DAG: ![[IA:[0-9]+]] = distinct !DILocation(line: 30, column: 1, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[TILE:[0-9]+]], ![[GPU:[0-9]+]])
; CHECK-DAG: ![[TILE]] = !DILayerLoc(line: 100, column: 1, file: !{{[0-9]+}}, kind: "tile ir")
; CHECK-DAG: ![[GPU]] = !DILayerLoc(line: 7, column: 3, file: !{{[0-9]+}}, kind: "gpu ir")

;; Both layered locations reference that one list node.
; CHECK-DAG: = distinct !DILocation(line: 40, column: 1, scope: !{{[0-9]+}}, irlayers: ![[LIST]])

;; The instruction locations reference them as inlinedAt and carry no layers of
;; their own.
; CHECK-DAG: !DILocation(line: 21, column: 5, scope: !{{[0-9]+}}, inlinedAt: ![[IA]])
