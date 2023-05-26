
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t %s 
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

; CHECK:        Address            Line   Column File   ISA Discriminator Flags
; CHECK-NEXT:   ------------------ ------ ------ ------ --- ------------- -------------
; CHECK-NEXT:   0x0000000000000000      3      0      1   0             0  is_stmt prologue_end
; CHECK-NEXT:   0x000000000000001c      3      0      1   0             0  is_stmt end_sequence

define i32 @main() !dbg !7 {
entry:
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "1.c", directory: "./")
!2 = !{i32 7, !"Dwarf Version", i32 3}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocation(line: 3, scope: !7)
