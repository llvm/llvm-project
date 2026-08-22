; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;; TODO: Enable once PTXAS changes land.
;; RUN-TODO: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
;;
;; Test that llvm.intermediate_level_source metadata generates .nv_intermediate_source_section
;; section in PTX output. This section contains the high-level source code from intermediate
;; representations like TileIR.
;;
;; This test uses tuple-style !dbg metadata where the first element is a DILocation
;; and the second element is TileIR metadata referencing a higher-level intermediate source
;; location.

;; Check that .file directives are emitted for the intermediate source files
; CHECK-DAG: .file [[FILE123:[0-9]+]] "{{.*}}c7b5df928863df5843aa56a544ebd647"
; CHECK-DAG: .file [[FILE456:[0-9]+]] "{{.*}}79645c2c541f9f72c475b950d40a107b"

;; Check that the intermediate source section is emitted
; CHECK: .nv_intermediate_source_section {

;; First code block with TileIR - sourceFileName matches the file number
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "TileIR"
; CHECK-NEXT:     .sourceFileName: [[FILE123]]
; CHECK-NEXT:     .source: <<<

;; Second code block with TileIR - sourceFileName matches the file number
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "TileIR"
; CHECK-NEXT:     .sourceFileName: [[FILE456]]
; CHECK-NEXT:     .source: <<<

;; Close the section
; CHECK: }

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

;; DILocations for the source file (test.cu)
!11 = !DILocation(line: 2, column: 5, scope: !8)
!12 = !DILocation(line: 3, column: 5, scope: !8)
!13 = !DILocation(line: 4, column: 5, scope: !8)
!14 = !DILocation(line: 5, column: 1, scope: !8)

;; High-level source file reference for tileIR_source.123
!15 = !DIFile(filename: "tileIR_source.123", directory: ".")
!16 = !DILexicalBlockFile(scope: !8, file: !15, discriminator: 0)
!17 = !DILocation(line: 100, column: 10, scope: !16)

;; TileIR metadata tuple - references the high-level source location (tileIR_source.123)
!18 = !{!"TileIR", !17}

;; High-level source file reference for tileIR_source.456
!24 = !DIFile(filename: "tileIR_source.456", directory: ".")
!25 = !DILexicalBlockFile(scope: !8, file: !24, discriminator: 0)
!26 = !DILocation(line: 200, column: 15, scope: !25)

;; TileIR metadata tuple - references the high-level source location (tileIR_source.456)
!27 = !{!"TileIR", !26}

;; Tuple metadata for instructions: {DILocation, TileIR_metadata}
;; First two instructions reference tileIR_source.123
!20 = !{!11, !18}
!21 = !{!12, !18}
;; Last two instructions reference tileIR_source.456
!22 = !{!13, !27}
!23 = !{!14, !27}

;; Intermediate-level source metadata - this is used to generates .nv_intermediate_source_section 
;; PTX section.
!llvm.intermediate_level_source = !{!30, !31}
!30 = !{!"TileIR", !15, !"%0 = memref.load %arg0[] : memref<f32>"}
!31 = !{!"TileIR", !24, !"memref.store %0, %arg1[] : memref<f32>"}
