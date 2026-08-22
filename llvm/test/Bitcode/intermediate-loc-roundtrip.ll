; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

;; Test that intermediate location metadata (used for multi-level line info)
;; round-trips correctly through bitcode.

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

;; Verify the metadata structure is preserved:
;; Each !dbg reference should point to a tuple containing a DILocation and
;; an intermediate location tuple with the kind string and DILocation.

; CHECK-DAG: ![[DBG1]] = !{![[SRC1:[0-9]+]], ![[INT:[0-9]+]]}
; CHECK-DAG: ![[DBG2]] = !{![[SRC2:[0-9]+]], ![[INT]]}
; CHECK-DAG: ![[DBG3]] = !{![[SRC3:[0-9]+]], ![[INT]]}

;; Check source locations
; CHECK-DAG: ![[SRC1]] = !DILocation(line: 2, column: 5, scope: ![[SP:[0-9]+]])
; CHECK-DAG: ![[SRC2]] = !DILocation(line: 3, column: 5, scope: ![[SP]])
; CHECK-DAG: ![[SRC3]] = !DILocation(line: 4, column: 1, scope: ![[SP]])

;; Check intermediate location tuple has kind string and DILocation
; CHECK-DAG: ![[INT]] = !{!"TileIR", ![[INTLOC:[0-9]+]]}
; CHECK-DAG: ![[INTLOC]] = !DILocation(line: 100, column: 10, scope: ![[INTSCOPE:[0-9]+]])

;; Check intermediate scope references intermediate file
; CHECK-DAG: ![[INTSCOPE]] = !DILexicalBlockFile(scope: ![[SP]], file: ![[INTFILE:[0-9]+]], discriminator: 0)
; CHECK-DAG: ![[INTFILE]] = !DIFile(filename: "intermediate.tileir", directory: ".")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
;; Required by the IR verifier whenever an instruction carries an
;; intermediate-loc !dbg chain: every intermediate DIFile referenced by a
;; chain must be declared here.
!llvm.intermediate_level_source = !{!100}
!100 = !{!"TileIR", !14, !""}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Source locations
!11 = !DILocation(line: 2, column: 5, scope: !8)
!12 = !DILocation(line: 3, column: 5, scope: !8)
!13 = !DILocation(line: 4, column: 1, scope: !8)

;; Intermediate location file and scope
!14 = !DIFile(filename: "intermediate.tileir", directory: ".")
!15 = !DILexicalBlockFile(scope: !8, file: !14, discriminator: 0)
!16 = !DILocation(line: 100, column: 10, scope: !15)

;; Intermediate metadata tuple: {kind_string, location}
!17 = !{!"TileIR", !16}

;; Combined source + intermediate location tuples
!20 = !{!11, !17}
!21 = !{!12, !17}
!22 = !{!13, !17}
