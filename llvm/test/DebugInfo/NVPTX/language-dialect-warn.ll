; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda 2>&1 | FileCheck %s

; CHECK-COUNT-1: warning: unknown NVPTX language dialect '42' on DICompileUnit; expected 'DW_LLVM_LANG_DIALECT_simt' or 'DW_LLVM_LANG_DIALECT_tile'

define void @kernel_bogus() !dbg !5 {
  ret void, !dbg !10
}

define void @kernel_simt() !dbg !15 {
  ret void, !dbg !16
}

define void @kernel_tile() !dbg !17 {
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0, !1, !2}
!llvm.module.flags = !{!12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: 42)
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !19, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_simt)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_tile)
!3 = !DIFile(filename: "test.cu", directory: "/tmp")
!19 = !DIFile(filename: "test-simt.cu", directory: "/tmp")
!20 = !DIFile(filename: "test-tile.cu", directory: "/tmp")
!4 = !{}
!9 = !DISubroutineType(types: !4)
!5 = distinct !DISubprogram(name: "kernel_bogus", scope: !3, file: !3, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !4)
!10 = !DILocation(line: 1, column: 1, scope: !5)
!15 = distinct !DISubprogram(name: "kernel_simt", scope: !19, file: !19, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!16 = !DILocation(line: 1, column: 1, scope: !15)
!17 = distinct !DISubprogram(name: "kernel_tile", scope: !20, file: !20, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!18 = !DILocation(line: 1, column: 1, scope: !17)

!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
