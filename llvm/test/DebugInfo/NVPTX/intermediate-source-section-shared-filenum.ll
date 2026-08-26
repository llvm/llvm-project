; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; Two intermediate DIFiles with identical content under different filenames
;; collapse onto one emitted .file number (a sourced file is named by its
;; checksum digest). Check the source section emits exactly one .code_block
;; for that number.

;; Both layers resolve to the same .file number, so both .loc_intermediate
;; directives reference it, and only one .file directive is emitted for it.
; CHECK-DAG: .loc_intermediate [[F:[0-9]+]] 100 10
; CHECK-DAG: .loc_intermediate [[F]] 200 20
; CHECK-DAG: .file [[F]] ".{{/|\\\\}}cccccccccccccccccccccccccccccccc"

;; EXACTLY ONE code_block: the section closes immediately after it.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "tile ir"
; CHECK-NEXT: .sourceFileName: [[F]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: shared source line
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

;; Two DIFile nodes differing ONLY in filename -- same directory, same checksum,
;; same source -- so they are distinct metadata but name the same content.
!14 = !DIFile(filename: "a.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "cccccccccccccccccccccccccccccccc", source: "shared source line")
!15 = !DILayerLoc(line: 100, column: 10, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)

!24 = !DIFile(filename: "b.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "cccccccccccccccccccccccccccccccc", source: "shared source line")
!25 = !DILayerLoc(line: 200, column: 20, file: !24, kind: "tile ir")
!26 = !DILayerLocList(!25)

!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !16)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !26)
!22 = !DILocation(line: 4, column: 1, scope: !8)
