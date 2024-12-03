; RUN:  opt -S -passes=globalopt --experimental-debuginfo-iterators=false <%s | FileCheck %s
; CHECK: #dbg_value(ptr undef, 
; CHECK-SAME:    [[VAR:![0-9]+]],
; CHECK-SAME:    !DIExpression()
; CHECK: [[VAR]] = !DILocalVariable(name: "_format"


; ModuleID = '<stdin>'
source_filename = "test.cpp"

@_ZZZZ4main_format = internal constant [24 x i8] c"Result1: Hello, World!\0A\00", align 16, !dbg !9

define void  @foo() align 2 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr @_ZZZZ4main_format, metadata !11, metadata !DIExpression()), !dbg !12
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/path/to")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 27, type: !6, scopeLine: 27, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "_format", scope: !5, file: !1, line: 49, type: !8, isLocal: true, isDefinition: true)
!11 = !DILocalVariable(name: "_format", arg: 1, scope: !5, file: !1, line: 79, type: !8)
!12 = !DILocation(line: 0, scope: !5)
