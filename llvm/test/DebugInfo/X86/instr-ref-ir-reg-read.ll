; RUN: llc %s -start-after=codegenprepare -stop-before=finalize-isel \
; RUN:    -experimental-debug-variable-locations=true  -o - \
; RUN: | FileCheck %s

; Test that the given input doesn't crash with instrruction referencing variable
; locations. The use of llvm.read_register allows the IR to access any register
; at any point, which is unfortunate, but a use case that needs to be supported.
;
; Just examine to see that we read something from $rsp.
; CHECK-LABEL: bb.1.if.then:
; CHECK:       DBG_PHI $rsp, 1
; CHECK:       DBG_INSTR_REF 1, 0

source_filename = "tlb-9e7172.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-cros-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: noredzone nounwind null_pointer_is_valid optsize sspstrong
define dso_local void @switch_mm_irqs_off() local_unnamed_addr #0 !dbg !16 {
entry:
  %0 = load i32, i32* @c, align 4, !dbg !24
  %tobool.not = icmp eq i32 %0, 0, !dbg !24
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !25

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !26
  %1 = tail call i64 @llvm.read_register.i64(metadata !7), !dbg !27
  call void @llvm.dbg.value(metadata i64 %1, metadata !20, metadata !DIExpression()), !dbg !26
  %call = tail call i32 @b(i64 noundef %1) #4, !dbg !28
  ret void, !dbg !29

if.end:                                           ; preds = %entry
  ret void, !dbg !29
}

; Function Attrs: nofree nounwind readonly
declare i64 @llvm.read_register.i64(metadata) #1

; Function Attrs: noredzone null_pointer_is_valid optsize
declare !dbg !30 dso_local i32 @b(i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!2}
!llvm.named.register.rsp = !{!7}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !5, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C89, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "tlb.c", directory: ".")
!4 = !{!0}
!5 = !DIFile(filename: "tlb-9e7172.c", directory: ".")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!"rsp"}
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 2}
!11 = !{i32 1, !"Code Model", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{i32 1, !"override-stack-alignment", i32 8}
!14 = !{i32 4, !"SkipRaxSetup", i32 1}
!15 = !{!"clang version 15.0.0 (git@github.com:llvm/llvm-project ab49dce01f211fd80f76f449035d771f5e2720b9)"}
!16 = distinct !DISubprogram(name: "switch_mm_irqs_off", scope: !5, file: !5, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!20}
!20 = !DILocalVariable(name: "d", scope: !21, file: !5, line: 6, type: !23)
!21 = distinct !DILexicalBlock(scope: !22, file: !5, line: 5, column: 10)
!22 = distinct !DILexicalBlock(scope: !16, file: !5, line: 5, column: 7)
!23 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!24 = !DILocation(line: 5, column: 7, scope: !22)
!25 = !DILocation(line: 5, column: 7, scope: !16)
!26 = !DILocation(line: 0, scope: !21)
!27 = !DILocation(line: 6, column: 14, scope: !21)
!28 = !DILocation(line: 7, column: 5, scope: !21)
!29 = !DILocation(line: 9, column: 1, scope: !16)
!30 = !DISubprogram(name: "b", scope: !5, file: !5, line: 3, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{!6, !23}
!33 = !{}
