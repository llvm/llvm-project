; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Two debug records sit between the merge instruction and its terminator, one
; value and one declare. NonSemantic.Shader.DebugInfo says DebugDeclare,
; DebugValue, DebugLine, DebugNoLine and DebugFunctionDefinition "cannot come
; after a 'Merge Instruction'", so both move ahead of the merge and nothing is
; left between the merge and the branch.
;
; A zero-operand fake use between the merge and the records becomes a
; non-debug meta instruction that emits no SPIR-V. It must not hide the merge
; when the records choose their output anchor.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[CONDNAME:%[0-9]+]] = OpString "condition"
; CHECK-DAG: [[SLOTNAME:%[0-9]+]] = OpString "slot"
; CHECK-DAG: [[CONDVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[CONDNAME]]
; CHECK-DAG: [[SLOTVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[SLOTNAME]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; The merge path emits both records while visiting the merge and relies on each
; record's own visit emitting nothing, so the counts are pinned: a second copy
; of either one has to fail.
; CHECK: [[SLOT:%[0-9]+]] = OpVariable
; CHECK: [[CMP:%[0-9]+]] = OpSLessThan
; CHECK-COUNT-1: OpExtInst [[VOID]] [[EXT]] DebugValue [[CONDVAR]] [[CMP]] [[EXPR]]{{ *$}}
; CHECK-NOT: DebugValue
; CHECK-COUNT-1: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[SLOTVAR]] [[SLOT]] [[EXPR]]{{ *$}}
; CHECK-NOT: DebugDeclare
; CHECK: OpSelectionMerge
; CHECK-NEXT: OpBranchConditional
; CHECK-NOT: DebugValue
; CHECK-NOT: DebugDeclare

target triple = "spirv64-unknown-unknown"

define spir_func i32 @if_else(i32 %x) !dbg !5 {
entry:
  %slot = alloca i32, align 4
  store i32 %x, ptr %slot, align 4
  %cmp = icmp slt i32 %x, 0, !dbg !8
  call void @llvm.spv.selection.merge.p0(ptr blockaddress(@if_else, %merge), i32 0), !dbg !14
  call void (...) @llvm.fake.use()
    #dbg_value(i1 %cmp, !16, !DIExpression(), !14)
    #dbg_declare(ptr %slot, !17, !DIExpression(), !14)
  br i1 %cmp, label %then, label %else, !dbg !13

then:
  br label %merge, !dbg !9

else:
  br label %merge, !dbg !10

merge:
  %r = load i32, ptr %slot, align 4, !dbg !12
  ret i32 %r, !dbg !12
}

declare void @llvm.spv.selection.merge.p0(ptr, i32 immarg)
declare void @llvm.fake.use(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-merge-region.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "if_else", linkageName: "if_else", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!9 = !DILocation(line: 4, column: 5, scope: !5)
!10 = !DILocation(line: 5, column: 5, scope: !5)
!12 = !DILocation(line: 9, column: 3, scope: !5)
!13 = !DILocation(line: 99, column: 50, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DIBasicType(name: "bool", size: 1, encoding: DW_ATE_boolean)
!16 = !DILocalVariable(name: "condition", scope: !5, file: !1, line: 7, type: !15)
!17 = !DILocalVariable(name: "slot", scope: !5, file: !1, line: 8, type: !7)
