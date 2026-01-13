; RUN: opt %s -o - -S | llvm-as - | llvm-dis - | FileCheck %s

; CHECK: !DIBasicType(name: "unsigned _BitInt", size: 32, dataSize: 17, encoding: DW_ATE_unsigned)

@a = global i8 0, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 22.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "bit-int.c", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "unsigned _BitInt", size: 32, dataSize: 17, encoding: DW_ATE_unsigned)
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 22.0.0git"}
