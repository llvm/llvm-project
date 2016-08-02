; RUN: llc %s -filetype=asm -o -

target datalayout = "E-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be--none-eabi"

@a = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "-", directory: "/work/validation")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !7, variable: i32* @a)
!5 = !DIFile(filename: "<stdin>", directory: "/work/validation")
!6 = !{!"<stdin>", !"/work/validation"}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.6.0 "}
