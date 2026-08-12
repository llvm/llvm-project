; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;; [TODO] Re-enable once ptxas changes have landed.
;; RUN-TODO: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
;;
;; Test that DILocation.irlayers (multi-level line info) generates
;; .loc_intermediate directives in PTX output: the primary source location is
;; the DILocation, and each DILayerLoc entry emits a secondary directive.
;; Instructions with only a source DILocation (no irlayers) do not emit one.

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .visible .func test_kernel

;; First instruction (store ptr) is layered.
; CHECK: .loc [[SRCFILE:[0-9]+]] 2 5
; CHECK-NEXT: .loc_intermediate [[INTFILE:[0-9]+]] 100 10

;; Second instruction (load ptr) has only a source DILocation.
; CHECK: .loc [[SRCFILE]] 3 5
; CHECK-NOT: .loc_intermediate

;; Third instruction (load i32) is layered.
; CHECK: .loc [[SRCFILE]] 5 5
; CHECK-NEXT: .loc_intermediate [[INTFILE]] 100 10

;; Fourth instruction (store i32) has only a source DILocation.
; CHECK: .loc [[SRCFILE]] 6 5
; CHECK-NOT: .loc_intermediate

;; Fifth instruction (ret) is layered.
; CHECK: .loc [[SRCFILE]] 4 1
; CHECK-NEXT: .loc_intermediate [[INTFILE]] 100 10

;; The .file declarations come after the function body.
; CHECK: .file [[SRCFILE]] "/test{{/|\\\\}}test.cu"
;; The secondary .file is named by the intermediate DIFile's checksum digest.
; CHECK: .file [[INTFILE]] ".{{/|\\\\}}0123456789abcdef0123456789abcdef"

define dso_local void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  %val = alloca i32, align 4
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  %0 = load ptr, ptr %v.addr, align 8, !dbg !12
  %1 = load i32, ptr %0, align 4, !dbg !23
  store i32 %1, ptr %val, align 4, !dbg !19
  ret void, !dbg !22
}

attributes #0 = { noinline optnone }

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
!14 = !DIFile(filename: "intermediate.ptx", directory: ".", checksumkind: CSK_MD5, checksum: "0123456789abcdef0123456789abcdef", source: "intermediate ptx text")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 100, column: 10, file: !14, kind: "TileIR")

;; Source-only instruction locations (no irlayers).
!12 = !DILocation(line: 3, column: 5, scope: !8)
!19 = !DILocation(line: 6, column: 5, scope: !8)

;; Layered instruction locations: primary source loc + irlayers.
!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !30)
!22 = !DILocation(line: 4, column: 1, scope: !8, irlayers: !30)
!23 = !DILocation(line: 5, column: 5, scope: !8, irlayers: !30)
