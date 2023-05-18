; RUN: llc -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK

; Regression test for function patching asserting in some cases when debug info activated.
; The code below reproduces this crash.

; Compilation flag:  clang -target x86_64-none-linux-gnu -c -O2 -g -fms-hotpatch patchable-prologue-debuginfo.c
; int func( int val ) {
;   int neg = -val;
;   return neg + 1;
; }

; CHECK: # -- Begin function func

; ModuleID = 'patchable-prologue-debuginfo.c'
source_filename = "patchable-prologue-debuginfo.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-none-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local i32 @func(i32 noundef %val) local_unnamed_addr #0 !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata i32 %val, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata !DIArgList(i32 0, i32 %val), metadata !15, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !16
  %add = sub i32 1, %val, !dbg !17
  ret i32 %add, !dbg !18
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "patchable-function"="prologue-short-redirect" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.4 (git@gitlab-ncsa.ubisoft.org:LLVM/llvm-project.git 17850fb41c5bddcd80a9c2714f7e293f49fa8bb2)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "patchable-prologue-debuginfo.c", directory: "D:\\saudi\\bugrepro-llvm-hotpatch-crash")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 15.0.4 (git@gitlab-ncsa.ubisoft.org:LLVM/llvm-project.git 17850fb41c5bddcd80a9c2714f7e293f49fa8bb2)"}
!9 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "val", arg: 1, scope: !9, file: !1, line: 1, type: !12)
!15 = !DILocalVariable(name: "neg", scope: !9, file: !1, line: 3, type: !12)
!16 = !DILocation(line: 0, scope: !9)
!17 = !DILocation(line: 4, column: 16, scope: !9)
!18 = !DILocation(line: 4, column: 5, scope: !9)
