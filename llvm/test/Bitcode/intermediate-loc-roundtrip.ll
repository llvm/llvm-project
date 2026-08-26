; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

;; Test that intermediate location metadata (used for multi-level line info)
;; round-trips correctly through bitcode. The layers live on the
;; DILocation's typed `irlayers` operand (a DILayerLocList of DILayerLoc
;; entries), so the whole DILocation/DILayerLocList/DILayerLoc/DIFile chain must
;; survive .ll -> .bc -> .ll.

define dso_local void @test_kernel(ptr noundef %v) !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  %0 = load ptr, ptr %v.addr, align 8, !dbg !21
  ret void, !dbg !22
}

; CHECK-LABEL: define dso_local void @test_kernel
; CHECK: store ptr %v, ptr %v.addr, align 8, !dbg ![[DBG1:[0-9]+]]
; CHECK: load ptr, ptr %v.addr, align 8, !dbg ![[DBG2:[0-9]+]]
; CHECK: ret void, !dbg ![[DBG3:[0-9]+]]

;; Verify the metadata structure is preserved: each !dbg is a DILocation whose
;; source line/col comes from the primary location and whose `irlayers` operand
;; points at a shared (uniqued) DILayerLocList.

; CHECK-DAG: ![[DBG1]] = !DILocation(line: 2, column: 5, scope: ![[SP:[0-9]+]], irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG: ![[DBG2]] = !DILocation(line: 3, column: 5, scope: ![[SP]], irlayers: ![[LIST]])
; CHECK-DAG: ![[DBG3]] = !DILocation(line: 4, column: 1, scope: ![[SP]], irlayers: ![[LIST]])

;; The shared layer list holds one DILayerLoc with the kind string and the
;; intermediate coordinate.
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-DAG: ![[LAYER]] = !DILayerLoc(line: 100, column: 10, file: ![[INTFILE:[0-9]+]], kind: "TileIR")
; CHECK-DAG: ![[INTFILE]] = !DIFile(filename: "intermediate.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "ffffffffffffffffffffffffffffffff")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Intermediate-IR layer: one DILayerLoc, shared (uniqued) across all layered
;; instructions via a single DILayerLocList.
!14 = !DIFile(filename: "intermediate.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "ffffffffffffffffffffffffffffffff")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 100, column: 10, file: !14, kind: "TileIR")

;; Layered instruction locations: primary source loc + shared irlayers.
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !30)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !30)
!22 = !DILocation(line: 4, column: 1, scope: !8, irlayers: !30)
