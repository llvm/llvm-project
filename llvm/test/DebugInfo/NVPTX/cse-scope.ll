; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | %ptxas-verify %}

; At -O0, two operations at the same line:column but with different
; DILexicalBlock scopes (e.g., one inside a `{ ... }` block, one outside)
; must remain distinct.  Covers the scope half of isSameSourceLocation.

define i32 @scopes(i32 %a, i32 %b) !dbg !3 {
; CHECK-LABEL: scopes(
; CHECK: add.s32
; CHECK: add.s32
  %x = add i32 %a, %b, !dbg !10
  %y = add i32 %a, %b, !dbg !11
  %z = add i32 %x, %y, !dbg !12
  ret i32 %z, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "scopes", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}

; Two distinct lexical blocks; both have the same file:line:column origin.
!6 = distinct !DILexicalBlock(scope: !3, file: !1, line: 2, column: 1)
!7 = distinct !DILexicalBlock(scope: !3, file: !1, line: 2, column: 1)

; %x at (2,1) inside the first block, %y at (2,1) inside the second.
!10 = !DILocation(line: 2, column: 1, scope: !6)
!11 = !DILocation(line: 2, column: 1, scope: !7)
!12 = !DILocation(line: 3, column: 1, scope: !3)
!13 = !DILocation(line: 4, column: 1, scope: !3)
