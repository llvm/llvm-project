; RUN: opt %s -verify-debuginfo-preserve -instcombine -disable-output 2>&1 | FileCheck --check-prefix=VERIFY %s

; VERIFY: CheckModuleDebugify (original debuginfo):

; RUN: opt %s -verify-each-debuginfo-preserve -O2 -disable-output 2>&1 | FileCheck --check-prefix=VERIFY-EACH %s

; VERIFY-EACH: DeadArgumentEliminationPass
; VERIFY-EACH: GlobalDCEPass

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo(i32 %i) !dbg !8 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load i32, i32* %i.addr, align 4, !dbg !15
  %call = call i32 @goo(i32 %0), !dbg !16
  ret i32 %call, !dbg !17
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local i32 @goo(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !DILocalVariable(name: "i", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = !DILocation(line: 2, column: 13, scope: !8)
!15 = !DILocation(line: 3, column: 14, scope: !8)
!16 = !DILocation(line: 3, column: 10, scope: !8)
!17 = !DILocation(line: 3, column: 3, scope: !8)
