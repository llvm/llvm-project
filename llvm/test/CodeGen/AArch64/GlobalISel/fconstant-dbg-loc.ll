; Make sure we don't assign debug locations to G_FCONSTANT(s) when lowering.

; RUN: llc -mtriple aarch64 -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck --match-full-lines %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define float @main() #0 !dbg !14 {
; CHECK: %0:_(s32) = G_FCONSTANT float 0.000000e+00
  ret float 0.000000e+00, !dbg !24
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, nameTableKind: GNU)
!3 = !DIFile(filename: "dbg.c", directory: "/")
!4 = !{}
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"PIC Level", i32 2}
!13 = !{!"clang"}
!14 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!24 = !DILocation(line: 7, column: 3, scope: !14)
