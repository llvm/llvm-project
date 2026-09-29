; RUN: llvm-dis -o - %s.bc | FileCheck %s

@g = common global i8 0, align 4, !dbg !0

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true, expr: !6)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang (llvm/trunk 288154)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "a.c", directory: "/")
!3 = !{}
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; Old-style DIExpression bitcode records using DW_OP_bit_piece should be
; upgraded to DW_OP_LLVM_fragment.
;
; CHECK: !DIExpression(DW_OP_LLVM_fragment, 8, 8)
!6 = !DIExpression(DW_OP_bit_piece, 8, 8)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
