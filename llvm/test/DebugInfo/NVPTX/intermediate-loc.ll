; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;; [TODO] Re-enable once ptxas changes have landed.
;; RUN-TODO: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
;;
;; Test that intermediate location metadata generates .loc_intermediate
;; directives in PTX output. This is used for multi-level line info
;; where we want to track both the original source location and the
;; intermediate representation location.

;; Also tests that instructions with only source DILocation (no intermediate)
;; do not emit a .loc_intermediate directive.

; CHECK: .target sm_{{[0-9]+}}, debug

; CHECK: .visible .func test_kernel

;; First instruction (store ptr) has intermediate location
; CHECK: .loc [[SRCFILE:[0-9]+]] 2 5
; CHECK-NEXT: .loc_intermediate [[INTFILE:[0-9]+]] 100 10

;; Second instruction (load ptr) has only source DILocation - no .loc_intermediate
; CHECK: .loc [[SRCFILE]] 3 5
; CHECK-NOT: .loc_intermediate

;; Third instruction (load i32) has intermediate location
; CHECK: .loc [[SRCFILE]] 5 5
; CHECK-NEXT: .loc_intermediate [[INTFILE]] 100 10

;; Fourth instruction (store i32) has only source DILocation - no .loc_intermediate
; CHECK: .loc [[SRCFILE]] 6 5
; CHECK-NOT: .loc_intermediate

;; Fifth instruction (ret) has intermediate location
; CHECK: .loc [[SRCFILE]] 4 1
; CHECK-NEXT: .loc_intermediate [[INTFILE]] 100 10

;; The .file declarations come after the function body
; CHECK: .file [[SRCFILE]] "/test{{/|\\\\}}test.cu"
; CHECK: .file [[INTFILE]] ".{{/|\\\\}}fbfefc15e49e689890f0733c8733fc4c"

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
!18 = !DILocation(line: 5, column: 5, scope: !8)
!19 = !DILocation(line: 6, column: 5, scope: !8)

;; Intermediate location file and scope
!14 = !DIFile(filename: "intermediate.ptx", directory: ".")
!15 = !DILexicalBlockFile(scope: !8, file: !14, discriminator: 0)
!16 = !DILocation(line: 100, column: 10, scope: !15)

;; Intermediate metadata tuple: {kind_string, location}
!17 = !{!"TileIR", !16}

;; Instruction debug locations:
!20 = !{!11, !17}
!22 = !{!13, !17}
!23 = !{!18, !17}
