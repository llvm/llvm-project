; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; Mixed intermediate files: one carries DIFile.source, the other only a checksum
;; (no source). A .loc_intermediate is only meaningful with the source it points
;; into -- the consumer rejects a reference to a file with no .code_block -- so
;; the source-less layer is dropped ENTIRELY: no .loc_intermediate, no .file
;; entry, no .code_block. The sourced layer is emitted as usual, named by its
;; checksum digest. Contrast intermediate-source-section-empty.ll, where EVERY
;; file is source-less and the section disappears with them.

; CHECK: .loc_intermediate [[FA:[0-9]+]] 100 10
; CHECK-NOT: .loc_intermediate
; CHECK: .file [[FA]] ".{{/|\\\\}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
; CHECK-NOT: .file

;; The section holds EXACTLY ONE code_block -- the sourced file's. It closes
;; right after it.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "tile ir"
; CHECK-NEXT: .sourceFileName: [[FA]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: aaa source line
; CHECK-NEXT: .source_end
; CHECK-NEXT: }
; CHECK-NEXT: }

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  store ptr null, ptr %v.addr, align 8, !dbg !21
  ret void, !dbg !22
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Sourced intermediate file (referenced first -> lower .file number).
!14 = !DIFile(filename: "aaa.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "aaa source line")
!15 = !DILayerLoc(line: 100, column: 10, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)

;; Source-less intermediate file (checksum only, no source:).
!24 = !DIFile(filename: "bbb.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
!25 = !DILayerLoc(line: 200, column: 20, file: !24, kind: "tile ir")
!26 = !DILayerLocList(!25)

;; First instruction references the sourced file; second the source-less file.
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !16)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !26)
!22 = !DILocation(line: 4, column: 1, scope: !8)
