; Check that DW_OP_lit<n> and DW_OP_xderef_size survive DWARF emission.
; DwarfExpression::addExpression() previously handled only DW_OP_lit0 and had
; no DW_OP_xderef_size case, so either would hit its llvm_unreachable default.
; Note that emitConstu() also encodes small DW_OP_constu operands as
; DW_OP_lit<n>, so the output cannot distinguish the two; this checks that
; DW_OP_lit<n> in the IR is accepted and lowered rather than crashing.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s

; CHECK: DW_AT_name{{.*}}"lit"
; CHECK: DW_AT_location (DW_OP_addrx 0x0, DW_OP_lit31, DW_OP_plus)
; CHECK: DW_AT_name{{.*}}"xd"
; CHECK: DW_AT_location (DW_OP_addrx 0x1, DW_OP_lit2, DW_OP_swap, DW_OP_xderef_size 0x4)

; DW_OP_lit<n> DW_OP_stack_value is a constant, like DW_OP_constu n
; DW_OP_stack_value, so a global with no address still gets a value.
; CHECK: DW_AT_name{{.*}}"litconst"
; CHECK: DW_AT_const_value (5)

@lit = global i32 0, align 4, !dbg !0
@xd = global i32 0, align 4, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_lit31, DW_OP_plus))
!1 = distinct !DIGlobalVariable(name: "lit", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "a.c", directory: "/")
!4 = !{!0, !5, !10}
!5 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef_size, 4))
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 2, !"Dwarf Version", i32 5}
!9 = distinct !DIGlobalVariable(name: "xd", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_lit5, DW_OP_stack_value))
!11 = distinct !DIGlobalVariable(name: "litconst", scope: !2, file: !3, line: 3, type: !6, isLocal: true, isDefinition: true)
