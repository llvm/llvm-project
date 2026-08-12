; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;; TODO: Enable once PTXAS changes land.
;; RUN-TODO: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
;;
;; Test that intermediate DIFile.source text generates the
;; .nv_intermediate_source_section in PTX output. This section contains the
;; high-level source code from intermediate representations like TileIR.
;;
;; The layers live on the DILocation's `irlayers` operand: each
;; instruction's DILocation carries a DILayerLocList of DILayerLoc entries that
;; reference the intermediate DIFile. The code_block for a declared source file
;; is emitted only when some instruction's layer references that file (see
;; intermediate-source-section-empty.ll for the skip case).

;; Check that .file directives are emitted for the intermediate source files.
;; The secondary .file name is the intermediate DIFile's carried checksum digest.
; CHECK-DAG: .file [[FILE123:[0-9]+]] "{{.*}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
; CHECK-DAG: .file [[FILE456:[0-9]+]] "{{.*}}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

;; Check that the intermediate source section is emitted
; CHECK: .nv_intermediate_source_section {

;; First code block with TileIR - sourceFileName matches the file number
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "TileIR"
; CHECK-NEXT:     .sourceFileName: [[FILE123]]
; CHECK-NEXT:     .source_begin
; CHECK-NEXT:     %0 = memref.load %arg0[] : memref<f32>
; CHECK-NEXT:     .source_end

;; Second code block with TileIR - sourceFileName matches the file number
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "TileIR"
; CHECK-NEXT:     .sourceFileName: [[FILE456]]
; CHECK-NEXT:     .source_begin
; CHECK-NEXT:     memref.store %0, %arg1[] : memref<f32>
; CHECK-NEXT:     .source_end

;; Close the second code_block, then the section itself.
; CHECK-NEXT: }
; CHECK-NEXT: }

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  %0 = load ptr, ptr %v.addr, align 8, !dbg !21
  store ptr %0, ptr %v.addr, align 8, !dbg !22
  ret void, !dbg !23
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

;; High-level source file reference for tileIR_source.123, with its shared layer.
!15 = !DIFile(filename: "tileIR_source.123", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "%0 = memref.load %arg0[] : memref<f32>")
!17 = !DILayerLoc(line: 100, column: 10, file: !15, kind: "TileIR")
!18 = !DILayerLocList(!17)

;; High-level source file reference for tileIR_source.456, with its shared layer.
!24 = !DIFile(filename: "tileIR_source.456", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "memref.store %0, %arg1[] : memref<f32>")
!26 = !DILayerLoc(line: 200, column: 15, file: !24, kind: "TileIR")
!27 = !DILayerLocList(!26)

;; Instruction locations: source DILocation + irlayers.
;; First two instructions reference tileIR_source.123
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !18)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !18)
;; Last two instructions reference tileIR_source.456
!22 = !DILocation(line: 4, column: 5, scope: !8, irlayers: !27)
!23 = !DILocation(line: 5, column: 1, scope: !8, irlayers: !27)
