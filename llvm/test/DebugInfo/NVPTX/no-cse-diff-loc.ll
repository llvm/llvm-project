; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | %ptxas-verify %}

; At -O0, NVPTX keys CSE by both DAG structure and DebugLoc.  Two identical
; operations at different source lines must each produce their own .loc entry
; and instruction rather than being folded into one.

define i32 @no_cse_diff_loc(i32 %a, i32 %b) !dbg !3 {
; CHECK-LABEL: no_cse_diff_loc(
; CHECK: .loc {{[0-9]+}} 1
; CHECK: add.s32
; CHECK: .loc {{[0-9]+}} 2
; CHECK: add.s32
  %x = add i32 %a, %b, !dbg !6
  %y = add i32 %a, %b, !dbg !7
  %z = add i32 %x, %y, !dbg !8
  ret i32 %z, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "no_cse_diff_loc", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 1, column: 1, scope: !3)
!7 = !DILocation(line: 2, column: 1, scope: !3)
!8 = !DILocation(line: 3, column: 1, scope: !3)
!9 = !DILocation(line: 4, column: 1, scope: !3)
