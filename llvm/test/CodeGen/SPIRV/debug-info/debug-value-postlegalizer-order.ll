; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info --stop-after=spirv-postlegalizer %s -o - | FileCheck %s

; SPIRVPostLegalizer::generateAssignType() must leave the ASSIGN_TYPE that takes
; over a definition ahead of any DBG_VALUE naming it. updateRegType() moves the
; insert point past debug records, so without pinning it back the ASSIGN_TYPE
; lands after the record and the record precedes its own definition.
;
; This is checked on Machine IR because the final SPIR-V cannot show it. The
; handler drops a record whose definition it has not reached, so a regression
; here removes the DebugValue from the module rather than misplacing it, and no
; output test would name what went missing.

; CHECK: %[[#SUM:]]:iid(s32) = G_ADD
; CHECK-NEXT: %[[#ASSIGNED:]]:iid(s32) = ASSIGN_TYPE %[[#SUM]](s32)
; CHECK-NEXT: DBG_VALUE %[[#ASSIGNED]](s32)

target triple = "spirv64-unknown-unknown"

define spir_func i32 @add_one(i32 %x) !dbg !5 {
entry:
  %sum = add i32 %x, %x, !dbg !10
    #dbg_value(i32 %sum, !9, !DIExpression(), !11)
  ret i32 %sum, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-postlegalizer-order.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "result", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 7, scope: !5)
!11 = !DILocation(line: 4, column: 3, scope: !5)
