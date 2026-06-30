; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; When !llvm.intermediate_level_source declares source content but no
;; instruction's !dbg chain references those intermediate files (so
;; FileToFileNum has no matching entry and every code_block is skipped),
;; NVPTXDwarfDebug::buildIntermediateSourceSection must return an empty
;; string so the PTX output omits the .nv_intermediate_source_section
;; entirely instead of emitting an empty stub.

;; A non-empty PTX is still produced...
; CHECK: .target sm_{{[0-9]+}}
;; ... but the section header must NOT appear anywhere in the output.
; CHECK-NOT: .nv_intermediate_source_section

define dso_local void @no_intermediate_loc_chains(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !11
  ret void, !dbg !12
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
;; Source content is declared, but no instruction's !dbg chain references
;; "tileIR_source.unused", so this entry is silently skipped at codegen.
!llvm.intermediate_level_source = !{!30}
!4 = !DIFile(filename: "tileIR_source.unused", directory: ".")
!30 = !{!"TileIR", !4, !"%0 = unused"}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "no_intermediate_loc_chains", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Plain DILocations only - no tuple-form chains.
!11 = !DILocation(line: 2, column: 5, scope: !8)
!12 = !DILocation(line: 4, column: 1, scope: !8)
