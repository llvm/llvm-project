;; A constant displacement from a global's address is a link-time constant too.
;; The displacement is folded into the DIExpression rather than carried as an
;; operand offset, so that it survives into a DBG_INSTR_REF, whose operands
;; cannot hold one. DebugInfo/MIR/X86/globaladdr-offset.mir covers the
;; operand-offset spelling that hand-written MIR and LiveDebugValues produce;
;; both reach the same DWARF.

; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -dwarf-version=5 \
; RUN:   -stop-after=livedebugvalues < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -dwarf-version=5 \
; RUN:   -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF

@g = global i64 0, align 8

;; Nothing in the function materializes the address, so without this the
;; variable is dropped outright.
; MIR-LABEL: name: globaladdr_offset{{$}}
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 8)
;
; DWARF-LABEL: DW_AT_name ("globaladdr_offset")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (DW_OP_addrx {{0x[0-9a-f]+}}, DW_OP_plus_uconst 0x8, DW_OP_stack_value)
; DWARF-NEXT: DW_AT_name ("x")
define void @globaladdr_offset() !dbg !5 {
entry:
    #dbg_value(ptr getelementptr (i8, ptr @g, i64 8), !7, !DIExpression(), !10)
  call void @sink(ptr null), !dbg !10
  ret void, !dbg !10
}

; MIR-LABEL: name: globaladdr_negative_offset{{$}}
; MIR: DBG_VALUE @g, $noreg, ![[#]], !DIExpression(DW_OP_constu, 8, DW_OP_minus)
;
; DWARF-LABEL: DW_AT_name ("globaladdr_negative_offset")
; DWARF: DW_TAG_variable
; DWARF-NEXT: DW_AT_location (DW_OP_addrx {{0x[0-9a-f]+}}, DW_OP_lit8, DW_OP_minus, DW_OP_stack_value)
; DWARF-NEXT: DW_AT_name ("y")
define void @globaladdr_negative_offset() !dbg !12 {
entry:
    #dbg_value(ptr getelementptr (i8, ptr @g, i64 -8), !13, !DIExpression(), !14)
  call void @sink(ptr null), !dbg !14
  ret void, !dbg !14
}

;; In a DBG_VALUE_LIST the displacement has to be applied to its own operand,
;; not to whatever happens to be on the stack. Which register the other operand
;; lands in is beside the point, so only the tail of the expression is matched.
; MIR-LABEL: name: globaladdr_offset_variadic{{$}}
; MIR: DBG_VALUE_LIST ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus_uconst, 8, DW_OP_plus, DW_OP_stack_value), $rdi, @g
;
; DWARF-LABEL: DW_AT_name ("globaladdr_offset_variadic")
; DWARF: DW_TAG_variable
; DWARF: DW_OP_addrx {{0x[0-9a-f]+}}, DW_OP_plus_uconst 0x8, DW_OP_plus, DW_OP_stack_value
; DWARF: DW_AT_name ("z")
define void @globaladdr_offset_variadic(i64 %n) !dbg !15 {
entry:
    #dbg_value(!DIArgList(i64 %n, ptr getelementptr (i8, ptr @g, i64 8)), !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !17)
  call void @sink_i64(i64 %n), !dbg !17
  ret void, !dbg !17
}

declare void @sink(ptr)
declare void @sink_i64(i64)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!3 = !DIFile(filename: "globaladdr-offset.c", directory: "/")
!4 = !DISubroutineType(types: !11)
!5 = distinct !DISubprogram(name: "globaladdr_offset", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !6)
!6 = !{!7}
!7 = !DILocalVariable(name: "x", scope: !5, file: !3, line: 2, type: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = !DILocation(line: 2, column: 1, scope: !5)
!11 = !{null}
!12 = distinct !DISubprogram(name: "globaladdr_negative_offset", scope: !3, file: !3, line: 4, type: !4, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !18)
!13 = !DILocalVariable(name: "y", scope: !12, file: !3, line: 5, type: !8)
!14 = !DILocation(line: 5, column: 1, scope: !12)
!15 = distinct !DISubprogram(name: "globaladdr_offset_variadic", scope: !3, file: !3, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!16 = !DILocalVariable(name: "z", scope: !15, file: !3, line: 8, type: !8)
!17 = !DILocation(line: 8, column: 1, scope: !15)
!18 = !{!13}
!19 = !{!16}
