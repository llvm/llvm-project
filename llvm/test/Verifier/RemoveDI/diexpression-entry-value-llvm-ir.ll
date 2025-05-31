; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

; CHECK-NOT: #dbg_value
; CHECK: Entry values are only allowed in MIR unless they target a swiftasync Argument
; CHECK: #dbg_value(i32 %param, !{{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1)
; CHECK-NOT: #dbg_value
; CHECK-NOT: Entry values are only allowed
; CHECK: warning: ignoring invalid debug info

define void @foo(i32 %param, ptr swiftasync %ok_param) !dbg !4 {
entry:
    #dbg_value(i32 %param, !8, !DIExpression(DW_OP_LLVM_entry_value, 1), !9)
    #dbg_value(ptr %ok_param, !8, !DIExpression(DW_OP_LLVM_entry_value, 1), !9)
    #dbg_value(ptr poison, !8, !DIExpression(DW_OP_LLVM_entry_value, 1), !9)
    #dbg_value(ptr undef, !8, !DIExpression(DW_OP_LLVM_entry_value, 1), !9)
  ret void
}


attributes #0 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, type: !5, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "param", arg: 1, scope: !4, file: !1, type: !7)
!9 = !DILocation(line: 0, scope: !4)
