; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

;; The source: text of each intermediate DIFile that a layer refers to is
;; emitted once, as a .code_block in .nv_intermediate_source_section, so a
;; consumer can show the intermediate IR without the file on disk. Each block
;; carries the layer's kind and the file's .file number.

;; Each intermediate file gets a .file named by its checksum.
; CHECK-DAG: .file [[FILE123:[0-9]+]] "{{.*}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
; CHECK-DAG: .file [[FILE456:[0-9]+]] "{{.*}}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

; CHECK: .nv_intermediate_source_section {

;; .sourceFileName is the file's .file number.
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "intermediate ir"
; CHECK-NEXT:     .sourceFileName: [[FILE123]]
; CHECK-NEXT:     .source_begin
; CHECK-NEXT:     %0 = memref.load %arg0[] : memref<f32>
; CHECK-NEXT:     .source_end

;; Likewise for the second file.
; CHECK:   .code_block {
; CHECK-NEXT:     .ir_name: "intermediate ir"
; CHECK-NEXT:     .sourceFileName: [[FILE456]]
; CHECK-NEXT:     .source_begin
; CHECK-NEXT:     memref.store %0, %arg1[] : memref<f32>
; CHECK-NEXT:     .source_end

;; The section closes after the second block.
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

;; First intermediate file and its layer.
!15 = !DIFile(filename: "first.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "%0 = memref.load %arg0[] : memref<f32>")
!17 = !DILayerLoc(line: 100, column: 10, file: !15, kind: "intermediate ir")
!18 = !DILayerLocList(!17)

;; Second intermediate file and its layer.
!24 = !DIFile(filename: "second.ir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "memref.store %0, %arg1[] : memref<f32>")
!26 = !DILayerLoc(line: 200, column: 15, file: !24, kind: "intermediate ir")
!27 = !DILayerLocList(!26)

;; The first two instructions use the first file, the last two the second.
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !18)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !18)
!22 = !DILocation(line: 4, column: 5, scope: !8, irlayers: !27)
!23 = !DILocation(line: 5, column: 1, scope: !8, irlayers: !27)
