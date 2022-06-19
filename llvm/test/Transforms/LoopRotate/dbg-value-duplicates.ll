; RUN: opt -S -loop-rotate -verify-memoryssa < %s | FileCheck %s
source_filename = "/tmp/loop.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Function Attrs: nounwind ssp uwtable
define void @f(float* %input, i64 %n, i64 %s) local_unnamed_addr #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata float* %input, metadata !15, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %n, metadata !16, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i64 %s, metadata !17, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !23
  ; CHECK:   call void @llvm.dbg.value(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !23
  ; CHECK-NOT:   call void @llvm.dbg.value(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !23
  br label %for.cond, !dbg !24

for.cond:                                         ; preds = %for.body, %entry
  ; CHECK: %i.02 = phi i64 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %i.0 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  call void @llvm.dbg.value(metadata i64 %i.0, metadata !18, metadata !DIExpression()), !dbg !23
  %cmp = icmp slt i64 %i.0, %n, !dbg !25
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !27

for.cond.cleanup:                                 ; preds = %for.cond
  ret void, !dbg !28

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %input, i64 %i.0, !dbg !29
  %0 = load float, float* %arrayidx, align 4, !dbg !29, !tbaa !30
  call void @bar(float %0), !dbg !34
  %add = add nsw i64 %i.0, %s, !dbg !35
  call void @llvm.dbg.value(metadata i64 %add, metadata !18, metadata !DIExpression()), !dbg !23
  ; CHECK:   call void @llvm.dbg.value(metadata i64 %add, metadata !18, metadata !DIExpression()), !dbg !23
  ; CHECK-NOT:   call void @llvm.dbg.value(metadata i64 %add, metadata !18, metadata !DIExpression()), !dbg !23
  br label %for.cond, !dbg !36, !llvm.loop !37
}

declare void @bar(float) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 316689) (llvm/trunk 316685)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/loop.c", directory: "/Data/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 316689) (llvm/trunk 316685)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !14)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !13, !13}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!14 = !{!15, !16, !17, !18}
!15 = !DILocalVariable(name: "input", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!16 = !DILocalVariable(name: "n", arg: 2, scope: !8, file: !1, line: 2, type: !13)
!17 = !DILocalVariable(name: "s", arg: 3, scope: !8, file: !1, line: 2, type: !13)
!18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 3, type: !13)
!19 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 3)
!20 = !DILocation(line: 2, column: 15, scope: !8)
!21 = !DILocation(line: 2, column: 32, scope: !8)
!22 = !DILocation(line: 2, column: 45, scope: !8)
!23 = !DILocation(line: 3, column: 18, scope: !19)
!24 = !DILocation(line: 3, column: 8, scope: !19)
!25 = !DILocation(line: 3, column: 26, scope: !26)
!26 = distinct !DILexicalBlock(scope: !19, file: !1, line: 3, column: 3)
!27 = !DILocation(line: 3, column: 3, scope: !19)
!28 = !DILocation(line: 5, column: 1, scope: !8)
!29 = !DILocation(line: 4, column: 9, scope: !26)
!30 = !{!31, !31, i64 0}
!31 = !{!"float", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !DILocation(line: 4, column: 5, scope: !26)
!35 = !DILocation(line: 3, column: 31, scope: !26)
!36 = !DILocation(line: 3, column: 3, scope: !26)
!37 = distinct !{!37, !27, !38}
!38 = !DILocation(line: 4, column: 17, scope: !19)
