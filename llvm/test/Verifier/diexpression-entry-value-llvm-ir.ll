; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s

; CHECK: Entry values are only allowed in MIR unless they target a swiftasync Argument
; CHECK: call void @llvm.dbg.value(metadata i32 %param, metadata !{{.*}}, metadata !DIExpression(DW_OP_LLVM_entry_value, 1))
; CHECK-NOT: llvm.dbg.value
; CHECK: warning: ignoring invalid debug info

define void @foo(i32 %param, ptr swiftasync %ok_param) !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32 %param, metadata !8, metadata !DIExpression(DW_OP_LLVM_entry_value, 1)), !dbg !9
  call void @llvm.dbg.value(metadata ptr %ok_param, metadata !8, metadata !DIExpression(DW_OP_LLVM_entry_value, 1)), !dbg !9
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #0

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
