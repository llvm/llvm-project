; RUN: llc -mtriple=arm64-apple-macosx11.0.0 -O2 -filetype=obj -o /dev/null < %s 2>&1 | FileCheck %s

; CHECK: #dbg record expression references nonexistent location operand
; CHECK: warning: ignoring invalid debug info

declare void @sink(i32)

define void @f(i32 %x, i32 %y) !dbg !4 {
entry:
    #dbg_value(i32 %x, !7, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !9)
  %s = add i32 %x, %y, !dbg !9
  call void @sink(i32 %s), !dbg !9
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DILocalVariable(name: "v", scope: !4, file: !1, line: 1, type: !6)
!9 = !DILocation(line: 1, column: 1, scope: !4)
