; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; The .code_blocks in .nv_intermediate_source_section are emitted in a
;; deterministic order: first-reference (insertion) order. The test pins that
;; the ordering key is reference order, not the filename -- "bbb.tileir" is
;; referenced first and "aaa.tileir" second, so bbb's code_block is emitted
;; first even though "aaa" sorts earlier alphabetically.

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  store ptr null, ptr %v.addr, align 8, !dbg !21
  ret void, !dbg !22
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

;; bbb is referenced first, so it receives the lower .file number.
; CHECK-DAG: .file [[FBBB:[0-9]+]] "{{.*}}22222222222222222222222222222222"
; CHECK-DAG: .file [[FAAA:[0-9]+]] "{{.*}}11111111111111111111111111111111"

; CHECK: .nv_intermediate_source_section {
;; "bbb.tileir" is referenced first -> lower file number -> emitted first.
; CHECK:      .code_block {
; CHECK-NEXT:   .ir_name: "tile ir"
; CHECK-NEXT:   .sourceFileName: [[FBBB]]
; CHECK-NEXT:   .source_begin
; CHECK-NEXT:   bbb source line
; CHECK-NEXT:   .source_end
;; "aaa.tileir" second (higher file number), despite sorting earlier by name.
; CHECK:      .code_block {
; CHECK-NEXT:   .ir_name: "tile ir"
; CHECK-NEXT:   .sourceFileName: [[FAAA]]
; CHECK-NEXT:   .source_begin
; CHECK-NEXT:   aaa source line
; CHECK-NEXT:   .source_end
; CHECK: }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; aaa.tileir: sorts first by filename; carries a distinct checksum + source.
!14 = !DIFile(filename: "aaa.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "11111111111111111111111111111111", source: "aaa source line")
!15 = !DILayerLoc(line: 10, column: 1, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)

;; bbb.tileir: sorts second by filename.
!24 = !DIFile(filename: "bbb.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "22222222222222222222222222222222", source: "bbb source line")
!25 = !DILayerLoc(line: 20, column: 1, file: !24, kind: "tile ir")
!26 = !DILayerLocList(!25)

;; First instruction references bbb, second references aaa (reverse of sort).
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !26)
!21 = !DILocation(line: 3, column: 5, scope: !8, irlayers: !16)
!22 = !DILocation(line: 4, column: 1, scope: !8)
