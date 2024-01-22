; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; RUN: llc < %s -O0 -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after livedebugvalues -o - | FileCheck %s -check-prefix=MIR

; RUN: llc --try-experimental-debuginfo-iterators < %s -mtriple=x86_64-- | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators < %s -O0 -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after livedebugvalues -o - | FileCheck %s -check-prefix=MIR

; PR4050

%0 = type { i64 }
%struct.S1 = type { i16, i32 }

@g_10 = external dso_local global %struct.S1, align 8

declare void @func_28(i64, i64)

; CHECK: movslq  g_10+4(%rip), %rdi
define void @int322(i32 %foo) !dbg !5 {
entry:
  %val = load i64, ptr @g_10, !dbg !16
  %0 = load i32, ptr getelementptr inbounds (%struct.S1, ptr @g_10, i32 0, i32 1), align 4, !dbg !17
; MIR: renamable {{\$r[a-z]+}} = MOVSX64rm32 {{.*}}, @g_10 + 4,{{.*}} debug-location !17 :: (dereferenceable load (s32) from @g_10 + 4)
  %1 = sext i32 %0 to i64, !dbg !18
  %tmp4.i = lshr i64 %val, 32, !dbg !19
  %tmp5.i = trunc i64 %tmp4.i to i32, !dbg !20
  %2 = sext i32 %tmp5.i to i64, !dbg !21
  tail call void @func_28(i64 %2, i64 %1) #0, !dbg !22
  call void @llvm.dbg.value(metadata i64 %val, metadata !8, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i32 %0, metadata !10, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i64 %1, metadata !12, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i64 %tmp4.i, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %tmp5.i, metadata !14, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %2, metadata !15, metadata !DIExpression()), !dbg !21
  ret void, !dbg !23
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/Users/vsk/src/llvm.org-main/llvm/test/CodeGen/X86/fold-sext-trunc.ll", directory: "/")
!2 = !{}
!3 = !{i32 8}
!4 = !{i32 6}
!5 = distinct !DISubprogram(name: "int322", linkageName: "int322", scope: null, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !2)
!7 = !{!8, !10, !12, !13, !14, !15}
!8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
!9 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 3, type: !9)
!13 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 4, type: !9)
!14 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 5, type: !11)
!15 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 6, type: !9)
!16 = !DILocation(line: 1, column: 1, scope: !5)
!17 = !DILocation(line: 2, column: 1, scope: !5)
!18 = !DILocation(line: 3, column: 1, scope: !5)
!19 = !DILocation(line: 4, column: 1, scope: !5)
!20 = !DILocation(line: 5, column: 1, scope: !5)
!21 = !DILocation(line: 6, column: 1, scope: !5)
!22 = !DILocation(line: 7, column: 1, scope: !5)
!23 = !DILocation(line: 8, column: 1, scope: !5)
!24 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!24}
