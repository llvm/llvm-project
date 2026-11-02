; RUN: llvm-as < %s | llvm-dis | FileCheck %s
;;
;; Checks that irlayers survive a bitcode round-trip on a DILocation written as
;; a METADATA_LOCATION record -- one reachable only as an inlinedAt target.

define dso_local void @test(ptr noundef %v) #0 !dbg !8 {
entry:
  store ptr %v, ptr %v, align 8, !dbg !20
  ret void, !dbg !21
}

attributes #0 = { noinline optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 20, type: !9, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0)

!14 = !DIFile(filename: "high-level.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "high-level ir text")
!15 = !DIFile(filename: "low-level.ir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "low-level ir text")

!16 = !DILayerLoc(line: 100, column: 1, file: !14, kind: "HighLevelIR")
!17 = !DILayerLoc(line: 7, column: 3, file: !15, kind: "LowLevelIR")
!18 = !DILayerLocList(!16, !17)

!19 = distinct !DILocation(line: 30, column: 1, scope: !8, irlayers: !18)
!20 = !DILocation(line: 21, column: 5, scope: !11, inlinedAt: !19)
!21 = !DILocation(line: 22, column: 1, scope: !11, inlinedAt: !19)

;; The metadata-block location keeps its layers, and the two entries keep their
;; order.
; CHECK-DAG: ![[IA:[0-9]+]] = distinct !DILocation(line: 30, column: 1, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[HIGH:[0-9]+]], ![[LOW:[0-9]+]])
; CHECK-DAG: ![[HIGH]] = !DILayerLoc(line: 100, column: 1, file: !{{[0-9]+}}, kind: "HighLevelIR")
; CHECK-DAG: ![[LOW]] = !DILayerLoc(line: 7, column: 3, file: !{{[0-9]+}}, kind: "LowLevelIR")

;; The instruction locations reference it as inlinedAt and carry no layers of
;; their own.
; CHECK-DAG: !DILocation(line: 21, column: 5, scope: !{{[0-9]+}}, inlinedAt: ![[IA]])
