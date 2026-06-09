; RUN: llc -mtriple=armv7-unknown-linux-gnueabihf < %s | FileCheck %s

; Ensure that debug instructions are not passed to stepBackward in load/store optimization
; which would lead to an assertion failure.


define i32 @_Unwind_VRS_Interpret(i64 %value) !dbg !4 {
; CHECK-LABEL: _Unwind_VRS_Interpret:
entry:
  %call4.i354 = call i32 (ptr, ptr, ...) @fprintf(ptr null, ptr null, ptr null, i32 0, i32 0, i32 0, i64 %value, i32 0)
    #dbg_value(i8 0, !7, !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value), !12)
  ret i32 0
}

declare i32 @fprintf(ptr, ptr, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "_Unwind_VRS_Interpret", scope: !5, file: !5, line: 260, type: !6, scopeLine: 261, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2, keyInstructions: true)
!5 = !DIFile(filename: "test.cpp", directory: "")
!6 = distinct !DISubroutineType(types: !2)
!7 = !DILocalVariable(name: "v", scope: !8, file: !5, line: 329, type: !9)
!8 = distinct !DILexicalBlock(scope: !4, file: !5, line: 326, column: 28)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !10, line: 51, baseType: !11)
!10 = !DIFile(filename: "stdint.h", directory: "")
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !DILocation(line: 0, scope: !8)
