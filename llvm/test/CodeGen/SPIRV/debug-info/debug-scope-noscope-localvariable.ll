; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info --print-after=spirv-nonsemantic-debug-info -O0 -mtriple=spirv64-unknown-unknown -stop-after=spirv-nonsemantic-debug-info  %s -o - | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefix=CHECK-OPTION
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-MIR: %[[SCOPE_ACT_1:[0-9]+]]:id(s32) = OpExtInst {{%[0-9]+}}(s64), 3, 23, {{%[0-9]+}}(s32)
; CHECK-MIR: %[[SCOPE_ACT_2:[0-9]+]]:id(s32) = OpExtInst {{%[0-9]+}}(s64), 3, 24
; CHECK-MIR: %[[VOID_TYPE:[0-9]+]]:type(s64) = OpTypeVoid
; CHECK-MIR: %[[STR:[0-9]+]]:id(s32) = OpString 7303014
; CHECK-MIR: %[[MD_FUNC:[0-9]+]]:id(s32) = OpExtInst %[[VOID_TYPE]](s64), 3, 20, %[[STR]](s32)
; CHECK-MIR: %[[STR_A:[0-9]+]]:id(s32) = OpString 97
; CHECK-MIR: %[[VAR_A:[0-9]+]]:id(s32) = OpExtInst %[[VOID_TYPE]](s64), 3, 26, %[[STR_A]](s32), {{%[0-9]+}}(s32), {{%[0-9]+}}(s32), {{%[0-9]+}}, {{%[0-9]+}}, %[[MD_FUNC]](s32)

; CHECK-SPIRV: %[[STR_FOO:[0-9]+]] = OpString "foo"
; CHECK-SPIRV: %[[STR_VAR_A:[0-9]+]] = OpString "a"
; CHECK-SPIRV: %[[DBG_FUNC_MD:[0-9]+]] = OpExtInst {{%[0-9]+}} %[[#]] DebugFunction %[[#]]
; CHECK-SPIRV: OpExtInst {{%[0-9]+}} %[[#]] DebugLocalVariable %[[STR_VAR_A]] {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} %[[DBG_FUNC_MD]]
; CHECK-SPIRV: OpExtInst {{%[0-9]+}} %[[#]] DebugScope %[[DBG_FUNC_MD]]
; CHECK-SPIRV: OpExtInst {{%[0-9]+}} %[[#]] DebugNoScope

; CHECK-OPTION-NOT: DebugScope
; CHECK-OPTION-NOT: DebugNoScope
; CHECK-OPTION-NOT: DebugLocalVariable

define dso_local i32 @foo(i32 noundef %0) local_unnamed_addr !dbg !10 {
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !15, metadata !DIExpression()), !dbg !17
  %2 = add nsw i32 %0, 1, !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %2, metadata !16, metadata !DIExpression()), !dbg !17
  ret i32 %2, !dbg !19
}

define dso_local i32 @bar(i32 noundef %0) local_unnamed_addr !dbg !20 {
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !22, metadata !DIExpression()), !dbg !24
  %2 = shl nsw i32 %0, 1, !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %2, metadata !23, metadata !DIExpression()), !dbg !24
  ret i32 %2, !dbg !26
}

define dso_local noundef i32 @main() local_unnamed_addr !dbg !27 {
  tail call void @llvm.dbg.value(metadata i32 4, metadata !31, metadata !DIExpression()), !dbg !33
  tail call void @llvm.dbg.value(metadata i32 8, metadata !32, metadata !DIExpression()), !dbg !33
  ret i32 12, !dbg !34
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "Ubuntu clang version 18.1.3 (1ubuntu1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/user/Ebin/llvm-project/llvm/test/CodeGen/SPIRV/debug-info", checksumkind: CSK_MD5, checksum: "f10967bf988e0df2d90961fe265242d5")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 2, type: !13)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 2, column: 13, scope: !10)
!19 = !DILocation(line: 3, column: 3, scope: !10)
!20 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "b", arg: 1, scope: !20, file: !1, line: 6, type: !13)
!23 = !DILocalVariable(name: "y", scope: !20, file: !1, line: 7, type: !13)
!24 = !DILocation(line: 0, scope: !20)
!25 = !DILocation(line: 7, column: 13, scope: !20)
!26 = !DILocation(line: 8, column: 3, scope: !20)
!27 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 11, type: !28, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{!13}
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "v1", scope: !27, file: !1, line: 12, type: !13)
!32 = !DILocalVariable(name: "v2", scope: !27, file: !1, line: 13, type: !13)
!33 = !DILocation(line: 0, scope: !27)
!34 = !DILocation(line: 14, column: 3, scope: !27)
