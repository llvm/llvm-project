; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: #dbg record expression references nonexistent location operand
; CHECK-NEXT: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)
; CHECK: #dbg record expression references nonexistent location operand
; CHECK-NEXT: #dbg_value(!DIArgList(i32 %a, i32 %b), ![[#]], !DIExpression(DW_OP_LLVM_arg, 2, DW_OP_stack_value)
; CHECK: #dbg record expression references nonexistent location operand
; CHECK-NEXT: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_LLVM_arg, 18446744073709551615, DW_OP_stack_value)
; CHECK: #dbg record expression references nonexistent location operand
; CHECK-NEXT: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_LLVM_arg, 4294967296, DW_OP_stack_value)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression(4101)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression(4101, 0, 1, 159)
; CHECK-NOT: #dbg record expression references nonexistent location operand
; CHECK-NOT: invalid expression
; CHECK: warning: ignoring invalid debug info

define void @single_location(i32 %a) !dbg !4 {
entry:
    #dbg_value(i32 %a, !7, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !9)
  ret void, !dbg !9
}

define void @arglist(i32 %a, i32 %b) !dbg !10 {
entry:
    #dbg_value(!DIArgList(i32 %a, i32 %b), !11, !DIExpression(DW_OP_LLVM_arg, 2, DW_OP_stack_value), !12)
    #dbg_value(!DIArgList(i32 %a, i32 %b), !11, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !12)
  ret void, !dbg !12
}

define void @wrapped_negative_index(i32 %a) !dbg !16 {
entry:
    #dbg_value(i32 %a, !17, !DIExpression(DW_OP_LLVM_arg, 18446744073709551615, DW_OP_stack_value), !18)
  ret void, !dbg !18
}

define void @index_truncating_to_zero(i32 %a) !dbg !19 {
entry:
    #dbg_value(i32 %a, !20, !DIExpression(DW_OP_LLVM_arg, 4294967296, DW_OP_stack_value), !21)
  ret void, !dbg !21
}

define void @missing_index(i32 %a) !dbg !22 {
entry:
    #dbg_value(i32 %a, !23, !DIExpression(DW_OP_LLVM_arg), !24)
  ret void, !dbg !24
}

define void @extra_index(i32 %a) !dbg !25 {
entry:
    #dbg_value(i32 %a, !26, !DIExpression(DW_OP_LLVM_arg, 0, 1, DW_OP_stack_value), !27)
  ret void, !dbg !27
}

define void @kill_location() !dbg !13 {
entry:
    #dbg_value(i32 poison, !14, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !15)
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "single_location", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DILocalVariable(name: "v", scope: !4, file: !1, line: 1, type: !6)
!9 = !DILocation(line: 1, column: 1, scope: !4)
!10 = distinct !DISubprogram(name: "arglist", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "w", scope: !10, file: !1, line: 2, type: !6)
!12 = !DILocation(line: 2, column: 1, scope: !10)
!13 = distinct !DISubprogram(name: "kill_location", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DILocalVariable(name: "u", scope: !13, file: !1, line: 3, type: !6)
!15 = !DILocation(line: 3, column: 1, scope: !13)
!16 = distinct !DISubprogram(name: "wrapped_negative_index", scope: !1, file: !1, line: 4, type: !5, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!17 = !DILocalVariable(name: "n", scope: !16, file: !1, line: 4, type: !6)
!18 = !DILocation(line: 4, column: 1, scope: !16)
!19 = distinct !DISubprogram(name: "index_truncating_to_zero", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DILocalVariable(name: "t", scope: !19, file: !1, line: 5, type: !6)
!21 = !DILocation(line: 5, column: 1, scope: !19)
!22 = distinct !DISubprogram(name: "missing_index", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!23 = !DILocalVariable(name: "m", scope: !22, file: !1, line: 6, type: !6)
!24 = !DILocation(line: 6, column: 1, scope: !22)
!25 = distinct !DISubprogram(name: "extra_index", scope: !1, file: !1, line: 7, type: !5, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!26 = !DILocalVariable(name: "e", scope: !25, file: !1, line: 7, type: !6)
!27 = !DILocation(line: 7, column: 1, scope: !25)
