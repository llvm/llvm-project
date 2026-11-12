; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-before=finalize-isel %s -o - | FileCheck %s
;
; Salvaging the truncation for %index adds DW_OP_LLVM_convert operations to
; operator_index's empty expression. It must also add DW_OP_stack_value:
; operator_index holds %index, not the contents of memory at address %index.
;
; CHECK: [[INDEX:![0-9]+]] = !DILocalVariable(name: "operator_index",
; CHECK: [[OFFSET:![0-9]+]] = !DILocalVariable(name: "offset",
; CHECK-LABEL: name: index_value
; CHECK: body:
; CHECK: DBG_INSTR_REF [[INDEX]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_stack_value)
; CHECK: DBG_INSTR_REF [[OFFSET]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_plus_uconst, 1)

declare void @observe(i64)

define i64 @index_value(i16 %code) !dbg !4 {
entry:
  %shift = lshr i16 %code, 7, !dbg !8
  %index = and i16 %shift, 3, !dbg !8
  #dbg_value(i16 %index, !7, !DIExpression(), !8)
  ; This artificial expression places offset in memory at address %index + 1.
  ; Salvaging must not add DW_OP_stack_value and turn that address into a value.
  #dbg_value(i16 %index, !10, !DIExpression(DW_OP_plus_uconst, 1), !8)
  %wide = zext i16 %index to i64, !dbg !8
  call void @observe(i64 %wide), !dbg !8
  ret i64 %wide, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "location.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "index_value", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !{!6, !9})
!6 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!7 = !DILocalVariable(name: "operator_index", scope: !4, file: !1, line: 2, type: !9)
!8 = !DILocation(line: 2, column: 1, scope: !4)
!9 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!10 = !DILocalVariable(name: "offset", scope: !4, file: !1, line: 2, type: !9)
