; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -fast-isel=false \
; RUN:   -dwarf-op-convert=Enable -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -fast-isel=true \
; RUN:   -dwarf-op-convert=Enable -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t

; Do not fold leading conversions when further operations depend on the
; constant's stack type. Keep the complete conversion chain for comparisons,
; including when the base type is as wide as the generic type.
; CHECK: [[SIGNED32:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name ("DW_ATE_signed_32")
; CHECK: [[SIGNED8:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name ("DW_ATE_signed_8")
; CHECK: [[SIGNED64:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name ("DW_ATE_signed_64")
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name ("test")
; CHECK: DW_AT_location ({{.*}}, DW_OP_convert ([[SIGNED32]]) "DW_ATE_signed_32", DW_OP_convert ([[SIGNED8]]) "DW_ATE_signed_8", DW_OP_consts +0, DW_OP_convert ([[SIGNED8]]) "DW_ATE_signed_8", DW_OP_lt, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("narrow")
; CHECK: DW_AT_location ({{.*}}, DW_OP_convert ([[SIGNED64]]) "DW_ATE_signed_64", DW_OP_consts +0, DW_OP_convert ([[SIGNED64]]) "DW_ATE_signed_64", DW_OP_lt, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("generic_width")

; Terminal conversions still fold away for a plain constant location.
; CHECK: DW_AT_location (DW_OP_lit2, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("constant")

@seen = global i8 0

define void @test() !dbg !4 {
  #dbg_value(i32 -1, !7, !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_consts, 0, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_lt, DW_OP_stack_value), !8)
  #dbg_value(i64 -1, !10, !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_consts, 0, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_lt, DW_OP_stack_value), !8)
  #dbg_value(i32 65538, !12, !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value), !8)
  store volatile i8 0, ptr @seen, !dbg !9
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "reduced", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !{!7, !10, !12})
!5 = !DISubroutineType(types: !{})
!6 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!7 = !DILocalVariable(name: "narrow", scope: !4, file: !1, line: 2, type: !6)
!8 = !DILocation(line: 2, column: 1, scope: !4)
!9 = !DILocation(line: 3, column: 1, scope: !4)
!10 = !DILocalVariable(name: "generic_width", scope: !4, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "constant", scope: !4, file: !1, line: 2, type: !13)
!13 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
