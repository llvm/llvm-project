; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4

define i32 @main() !dbg !4 {
entry:
  %0 = load volatile i32, i32* @x, align 4, !dbg !9, !tbaa !10
  %add = add nsw i32 %0, 24, !dbg !9
  store volatile i32 %add, i32* @x, align 4, !dbg !9, !tbaa !10
  %1 = load volatile i32, i32* @x, align 4, !dbg !14, !tbaa !10
  %add1 = add nsw i32 %1, 2, !dbg !14
  store volatile i32 %add1, i32* @x, align 4, !dbg !14, !tbaa !10
  %2 = load volatile i32, i32* @x, align 4, !dbg !15, !tbaa !10
  %add2 = add nsw i32 %2, 3, !dbg !15
  store volatile i32 %add2, i32* @x, align 4, !dbg !15, !tbaa !10
  %3 = load volatile i32, i32* @x, align 4, !dbg !16, !tbaa !10
  %add3 = add nsw i32 %3, 4, !dbg !16
  store volatile i32 %add3, i32* @x, align 4, !dbg !16, !tbaa !10
  tail call void @exit(i32 24), !dbg !17
  unreachable, !dbg !17
}

; CHECK-LABEL: main:
; CHECK:      .loc 1 3
; CHECK:      .loc 1 4
; CHECK:      .loc 1 5
; CHECK:      .loc 1 6
; CHECK:      .loc 1 7
; CHECK:      .loc 1 8
; CHECK-NEXT: movl  $24, %edi
; CHECK-NEXT: callq exit

declare void @exit(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 259383) (llvm/trunk 259385)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "t.c", directory: "/home/majnemer/llvm/src")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, variables: !2)
!5 = !DISubroutineType(types: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 4, column: 5, scope: !4)
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !DILocation(line: 5, column: 5, scope: !4)
!15 = !DILocation(line: 6, column: 5, scope: !4)
!16 = !DILocation(line: 7, column: 5, scope: !4)
!17 = !DILocation(line: 8, column: 3, scope: !4)
