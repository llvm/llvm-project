; RUN: llc -O0 -mtriple=aarch64-unknown-linux-gnu -global-isel -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 -mtriple=aarch64-unknown-linux-gnu -global-isel -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O2 -mtriple=aarch64-unknown-linux-gnu -global-isel -filetype=obj %s -o - \
; RUN:   | llvm-dwarfdump --name=sum --name=wsum - | FileCheck %s --check-prefix=DWARF

; Check that a variadic dbg_value lowered by GlobalISel survives as a
; DBG_VALUE_LIST through legalization, register bank selection and
; instruction selection, and produces a DWARF location.

; MIR-LABEL: name: f
; MIR:       [[A:%[0-9]+]]:gpr32 = COPY $w0
; MIR:       [[B:%[0-9]+]]:gpr32 = COPY $w1
; MIR:       DBG_VALUE_LIST ![[SUM:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), [[A]], [[B]]

; MIR-LABEL: name: g
; MIR:       [[X:%[0-9]+]]:gpr32 = COPY $w0
; MIR:       [[Y:%[0-9]+]]:gpr32 = COPY $w1
; MIR:       DBG_VALUE_LIST ![[WSUM:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_plus, DW_OP_stack_value), [[X]], [[Y]]

; DWARF:      DW_TAG_variable
; DWARF:        DW_AT_location
; DWARF-NEXT:     DW_OP_breg0 W0+0, DW_OP_breg1 W1+0, DW_OP_plus, DW_OP_stack_value
; DWARF:        DW_AT_name ("sum")
; DWARF:      DW_TAG_variable
; DWARF:        DW_AT_location
; DWARF-NEXT:     DW_OP_breg0 W0+0, DW_OP_convert {{.*}}, DW_OP_convert {{.*}}, DW_OP_breg1 W1+0, DW_OP_convert {{.*}}, DW_OP_convert {{.*}}, DW_OP_plus, DW_OP_stack_value
; DWARF:        DW_AT_name ("wsum")

define i32 @f(i32 %a, i32 %b) !dbg !5 {
entry:
    #dbg_value(!DIArgList(i32 %a, i32 %b), !9, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !10)
  %m = mul i32 %a, %b, !dbg !10
  ret i32 %m, !dbg !10
}

define i8 @g(i8 %x, i8 %y) !dbg !11 {
entry:
    #dbg_value(!DIArgList(i8 %x, i8 %y), !12, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !13)
  %m = mul i8 %x, %y, !dbg !13
  ret i8 %m, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{!4, !4, !4}
!8 = !{!9}
!9 = !DILocalVariable(name: "sum", scope: !5, file: !1, line: 2, type: !4)
!10 = !DILocation(line: 2, column: 3, scope: !5)
!11 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 5, type: !6, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!12 = !DILocalVariable(name: "wsum", scope: !11, file: !1, line: 6, type: !15)
!13 = !DILocation(line: 6, column: 3, scope: !11)
!14 = !{!12}
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
