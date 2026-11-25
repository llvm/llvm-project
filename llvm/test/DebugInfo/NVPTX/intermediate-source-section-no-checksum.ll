; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; An intermediate file with source but no checksum still needs a unique
;; secondary .file name, because ptxas keys the stored source by that name. The
;; name is then a hash of the DIFile's directory and filename. With a checksum,
;; the checksum is the name; see intermediate-checksum-secondary-file.ll.

;; The name is MD5(".intermediate.ir"): the directory followed by the filename.
; CHECK-DAG: .loc_intermediate [[F:[0-9]+]] 42 5
; CHECK-DAG: .file [[F]] ".{{/|\\\\}}b8d144e9353de1a3231a42ddb1305268"

;; The source still reaches the section.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "intermediate ir"
; CHECK-NEXT: .sourceFileName: [[F]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: intermediate ir source
; CHECK-NEXT: .source_end
; CHECK-NEXT: }
; CHECK-NEXT: }

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  ret void, !dbg !21
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

;; Source present, checksum absent.
!14 = !DIFile(filename: "intermediate.ir", directory: ".", source: "intermediate ir source")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 42, column: 5, file: !14, kind: "intermediate ir")

!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !30)
!21 = !DILocation(line: 3, column: 1, scope: !8)
