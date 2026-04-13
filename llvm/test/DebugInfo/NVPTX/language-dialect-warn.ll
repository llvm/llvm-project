; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda 2>&1 | FileCheck %s

; CHECK: warning: unknown NVPTX language dialect 'bogus' on DICompileUnit; expected 'simt' or 'tile'

define void @kernel() !dbg !5 {
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: "bogus")
!3 = !DIFile(filename: "test.cu", directory: "/tmp")
!4 = !{}
!9 = !DISubroutineType(types: !4)
!5 = distinct !DISubprogram(name: "kernel", scope: !3, file: !3, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !4)
!10 = !DILocation(line: 1, column: 1, scope: !5)

!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
