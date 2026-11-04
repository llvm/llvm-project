; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-streaming-hazard-size=0 \
; RUN:   -mattr=+sme -verify-machineinstrs < %s | FileCheck %s

; Debug instructions must not keep a redundant streaming-mode pair alive.

define void @remove_streaming_mode_pair_across_debug_instr() #0 !dbg !3 {
; CHECK-LABEL: remove_streaming_mode_pair_across_debug_instr:
; CHECK-NOT:     smstart
; CHECK-NOT:     smstop
; CHECK:         ret
  #dbg_value(i32 0, !6, !DIExpression(), !8)
  ret void
}

attributes #0 = { "aarch64_pstate_sm_body" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "sme-peephole-debug-instr.ll", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "remove_streaming_mode_pair_across_debug_instr", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !9)
!5 = !{!6}
!6 = !DILocalVariable(name: "value", scope: !3, file: !1, line: 1, type: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 1, column: 1, scope: !3)
!9 = !{null}
