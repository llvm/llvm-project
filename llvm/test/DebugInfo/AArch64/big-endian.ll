; RUN: llc %s -filetype=asm -o -

source_filename = "test/DebugInfo/AArch64/big-endian.ll"
target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be--none-eabi"

@a = common global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "<stdin>", directory: "/work/validation")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, producer: "clang version 3.6.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !7, imports: !6)
!5 = !DIFile(filename: "-", directory: "/work/validation")
!6 = !{}
!7 = !{!0}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.6.0 "}

