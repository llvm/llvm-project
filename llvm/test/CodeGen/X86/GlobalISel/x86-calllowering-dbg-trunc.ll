; RUN: llc -mtriple=i386-linux-gnu -global-isel -verify-machineinstrs < %s -o - | FileCheck %s --check-prefix=ALL

; This file is the output of clang -g -O2
; int test_dbg_trunc(unsigned long long a) { return a; }
;
; The intent of this check is to ensure the DBG_VALUE use of G_MERGE_VALUES is undef'd when the legalizer erases it.

; ModuleID = 'x86-calllowering-dbg-trunc.c'
source_filename = "x86-calllowering-dbg-trunc.c"
target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386"

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local i32 @test_dbg_trunc(i64 %a) local_unnamed_addr #0 !dbg !9 {
; ALL-LABEL: test_dbg_trunc:
; ALL:       # %bb.0: # %entry
; ALL:       pushl	%ebp
; ALL:       movl	%esp, %ebp
; ALL:       movl	8(%ebp), %eax
; ALL:       #DEBUG_VALUE: test_dbg_trunc:a <- undef
; ALL:       popl	%ebp
; ALL:       retl
entry:
  call void @llvm.dbg.value(metadata i64 %a, metadata !15, metadata !DIExpression()), !dbg !16
  %conv = trunc i64 %a to i32, !dbg !17
  ret i32 %conv, !dbg !18
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project ...)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "x86-calllowering-dbg-trunc.c", directory: "/tmp")
!2 = !{i32 1, !"NumRegisterParameters", i32 0}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project ...)"}
!9 = distinct !DISubprogram(name: "test_dbg_trunc", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!14 = !{!15}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 0, scope: !9)
!17 = !DILocation(line: 1, column: 51, scope: !9)
!18 = !DILocation(line: 1, column: 44, scope: !9)
