; RUN: llc -mtriple=riscv64 -relocation-model=static -O2 \
; RUN:   -stop-after=riscv-merge-base-offset -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s

; CHECK-LABEL: name: test
; CHECK: %[[BASE:[0-9]+]]:gpr = LUI target-flags(riscv-hi) @values + 8
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, !{{[0-9]+}},
; CHECK-SAME: !DIExpression(DW_OP_plus_uconst, 8, DW_OP_stack_value),
; CHECK-SAME: debug-location !{{[0-9]+}}
; CHECK-NEXT: %{{[0-9]+}}:gpr = LD killed %[[BASE]], target-flags(riscv-lo) @values + 8

@values = external hidden local_unnamed_addr global [2 x i64], align 8

define i64 @test() !dbg !4 {
entry:
    #dbg_value(ptr getelementptr inbounds (i8, ptr @values, i64 8), !8, !DIExpression(), !9)
  %value = load i64, ptr getelementptr inbounds (i8, ptr @values, i64 8), align 8, !dbg !10
  ret i64 %value, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "merge-base-offset-debug-use.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{!12}
!7 = !{!8}
!8 = !DILocalVariable(name: "p", scope: !4, file: !1, line: 2, type: !13)
!9 = !DILocation(line: 0, scope: !4)
!10 = !DILocation(line: 3, column: 10, scope: !4)
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
