; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --verify %t
; RUN: llvm-dwarfdump --debug-names %t | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -dwarf64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --verify %t
; RUN: llvm-dwarfdump --debug-names %t | FileCheck %s

;; Base types synthesized for DW_OP_convert must be indexed by name.

; CHECK:     .debug_names contents:
; CHECK-DAG: String: {{.*}} "DW_ATE_unsigned_16"
; CHECK-DAG: String: {{.*}} "DW_ATE_unsigned_64"

define i64 @convert(i16 %value) !dbg !4 {
  #dbg_value(i16 %value, !7, !DIExpression(DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_stack_value), !8)
  %wide = zext i16 %value to i64, !dbg !8
  ret i64 %wide, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: Default)
!1 = !DIFile(filename: "convert.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "convert", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, unit: !0, spFlags: DISPFlagDefinition | DISPFlagOptimized)
!5 = !DISubroutineType(types: !6)
!6 = !{!9, !10}
!7 = !DILocalVariable(name: "wide", scope: !4, file: !1, line: 1, type: !9)
!8 = !DILocation(line: 1, column: 1, scope: !4)
!9 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!10 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
