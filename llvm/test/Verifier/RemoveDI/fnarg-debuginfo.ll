; RUN: llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s


define void @foo() !dbg !2 {
entry:
  %a = alloca i32
  ; CHECK: conflicting debug info for argument
    #dbg_value(i32 0, !3, !DIExpression(), !6)
    #dbg_declare(ptr %a, !4, !DIExpression(), !6)
  ret void, !dbg !6
}

; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = distinct !DISubprogram(name: "foo", scope: !0, isDefinition: true, unit: !0)
!3 = !DILocalVariable(name: "a", arg: 1, scope: !2, file: !1, line: 1, type: !5)
!4 = !DILocalVariable(name: "b", arg: 1, scope: !2, file: !1, line: 1, type: !5)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DILocation(line: 1, scope: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
