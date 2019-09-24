; RUN: opt %s -indvars -verify -S -o - | FileCheck %s

; Hand-reduced from this example:
;
; #include <stdio.h>

; int main( int argc, char **argv )
; {
;    for( int ArgIndex = 1; ArgIndex < argc; ArgIndex += 1 )
;    {
;       printf("\n Argument %d:  %s\n", ArgIndex, argv[ArgIndex] );
;    }
; }

; clang++ -g -O -mllvm -disable-llvm-optzns -gno-column-info
; opt  -mem2reg -loop-rotate -scalar-evolution

; CHECK: @main
; CHECK: llvm.dbg.value(metadata i32 1, metadata [[METADATA_IDX1:![0-9]+]]
; CHECK: %[[VAR_NAME:.*]] = add nuw nsw i64
; CHECK: llvm.dbg.value(metadata i64 %[[VAR_NAME]], metadata [[METADATA_IDX1]], metadata !DIExpression())
; CHECK: DICompileUnit
; CHECK: [[METADATA_IDX1]] = !DILocalVariable(name: "ArgIndex"

source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [20 x i8] c"\0A Argument %d:  %s\0A\00", align 1

define dso_local i32 @main(i32 %argc, i8** %argv) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 1, metadata !17, metadata !DIExpression()), !dbg !19
  %cmp1 = icmp slt i32 1, %argc, !dbg !19
  br i1 %cmp1, label %for.body.lr.ph, label %for.cond.cleanup, !dbg !19

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !19

for.cond.for.cond.cleanup_crit_edge:              ; preds = %for.inc
  br label %for.cond.cleanup, !dbg !19

for.cond.cleanup:                                 ; preds = %for.cond.for.cond.cleanup_crit_edge, %entry
  br label %for.end

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %ArgIndex.02 = phi i32 [ 1, %for.body.lr.ph ], [ %add, %for.inc ]
  call void @llvm.dbg.value(metadata i32 %ArgIndex.02, metadata !17, metadata !DIExpression()), !dbg !19
  %idxprom = sext i32 %ArgIndex.02 to i64, !dbg !19
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 %idxprom, !dbg !19
  %0 = load i8*, i8** %arrayidx, align 8, !dbg !19
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str, i64 0, i64 0), i32 %ArgIndex.02, i8* %0), !dbg !19
  br label %for.inc, !dbg !19

for.inc:                                          ; preds = %for.body
  %add = add nsw i32 %ArgIndex.02, 1, !dbg !19
  call void @llvm.dbg.value(metadata i32 %add, metadata !17, metadata !DIExpression()), !dbg !19
  %cmp = icmp slt i32 %add, %argc, !dbg !19
  br i1 %cmp, label %for.body, label %for.cond.for.cond.cleanup_crit_edge, !dbg !19, !llvm.loop !19

for.end:                                          ; preds = %for.cond.cleanup
  ret i32 0, !dbg !19
}

declare dso_local i32 @printf(i8*, ...)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{!15, !16, !17}
!15 = !DILocalVariable(name: "argc", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!16 = !DILocalVariable(name: "argv", arg: 2, scope: !7, file: !1, line: 4, type: !11)
!17 = !DILocalVariable(name: "ArgIndex", scope: !18, file: !1, line: 6, type: !10)
!18 = distinct !DILexicalBlock(scope: !7, file: !1, line: 6)
!19 = !DILocation(line: 0, scope: !7)
