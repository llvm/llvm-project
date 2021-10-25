; RUN: opt < %s -passes=mem2reg -S | FileCheck %s
source_filename = "bugpoint-output.bc"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @scan() #0 !dbg !12 {
entry:
  %entry1 = alloca i8, align 8
  call void @llvm.dbg.declare(metadata i8* %entry1, metadata !18, metadata !19), !dbg !20
  store i8 0, i8* %entry1, align 8, !dbg !20
  br label %for.cond, !dbg !20

for.cond:
; CHECK: %[[PHI:.*]] = phi i8 [ 0, %entry ], [ %0, %for.cond ]
  %entryN = load i8, i8* %entry1, align 8, !dbg !20
; CHECK: call void @llvm.dbg.value(metadata i8 %[[PHI]],
; CHECK-SAME:                      metadata !DIExpression())
  %0 = add i8 %entryN, 1
; CHECK: %0 = add i8 %[[PHI]], 1
; CHECK: call void @llvm.dbg.value(metadata i8 %0,
; CHECK-SAME:                      metadata !DIExpression())
  store i8 %0, i8* %entry1, align 8, !dbg !20
  br label %for.cond, !dbg !20
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "adrian", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !13, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !4, !4}
!15 = !{!18}
!18 = !DILocalVariable(name: "entry", scope: !12, file: !1, line: 6, type: !4)
!19 = !DIExpression()
!20 = !DILocation(line: 6, scope: !12)
