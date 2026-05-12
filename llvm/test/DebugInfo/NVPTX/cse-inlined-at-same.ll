; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | %ptxas-verify %}

; Negative control for cse-inlined-at.ll: two operations with identical
; line:column:scope:inlinedAt must still CSE at -O0.  We're not over-splitting
; the bucket just because inlinedAt is present.

define i32 @caller(i32 %a, i32 %b) !dbg !3 {
; CHECK-LABEL: caller(
; CHECK:     add.s32 [[REG:%r[0-9]+]],
; CHECK-NOT: add.s32 {{%r[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK:     add.s32 {{%r[0-9]+}}, [[REG]], [[REG]]
  %x = add i32 %a, %b, !dbg !20
  %y = add i32 %a, %b, !dbg !20
  %z = add i32 %x, %y, !dbg !10
  ret i32 %z, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}

!3 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!10 = !DILocation(line: 5, column: 1, scope: !3)
!11 = !DILocation(line: 6, column: 1, scope: !3)

!12 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 10, type: !4, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!20 = !DILocation(line: 10, column: 1, scope: !12, inlinedAt: !21)
!21 = distinct !DILocation(line: 2, column: 1, scope: !3)
