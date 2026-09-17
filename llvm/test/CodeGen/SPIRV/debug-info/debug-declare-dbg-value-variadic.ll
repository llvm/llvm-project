; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugDeclare --implicit-check-not=DebugExpression
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A variadic #dbg_value. IRTranslator cannot lower the DIArgList.
; It emits DBG_VALUE $noreg, 0, which is an indirect DBG_VALUE just like a
; declare is.
; The test checks that it must not become a DebugDeclare.

; Clang seem to emit no DIArgList. The optimizer does,
; when it deletes a dead binary operation and rewrites the variable as an
; expression over the two operands.

; CHECK: OpExtInst {{.*}} DebugLocalVariable

target triple = "spirv64-unknown-unknown"

define spir_func i32 @sum(i32 %a, i32 %b) !dbg !5 {
entry:
  %add = add nsw i32 %a, %b, !dbg !11
    #dbg_value(!DIArgList(i32 %a, i32 %b), !9, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !11)
  ret i32 %add, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-dbg-value-variadic.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "sum", linkageName: "sum", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "total", scope: !5, file: !1, line: 2, type: !7)
!11 = !DILocation(line: 2, column: 7, scope: !5)
