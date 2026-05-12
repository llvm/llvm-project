; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | %ptxas-verify %}

; At -O0, NVPTX keys CSE by both DAG structure and DebugLoc.  Two identical
; operations at the SAME source location must be folded into one instruction;
; the consumer of both results must therefore see the same register for both
; operands.

define i32 @cse_same_loc(i32 %a, i32 %b) !dbg !3 {
; CHECK-LABEL: cse_same_loc(
; CHECK:     .loc {{[0-9]+}} 1
; CHECK:     add.s32 [[REG:%r[0-9]+]],
; CHECK-NOT: .loc {{[0-9]+}} 1
; CHECK:     .loc {{[0-9]+}} 3
; CHECK:     add.s32 {{%r[0-9]+}}, [[REG]], [[REG]]
  %x = add i32 %a, %b, !dbg !6
  %y = add i32 %a, %b, !dbg !6
  %z = add i32 %x, %y, !dbg !8
  ret i32 %z, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "cse_same_loc", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 1, column: 1, scope: !3)
!8 = !DILocation(line: 3, column: 1, scope: !3)
!9 = !DILocation(line: 4, column: 1, scope: !3)
