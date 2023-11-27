; RUN: opt < %s --passes='print<source-expr>' -disable-output  2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: Load Store Expression foo
define dso_local void @foo(ptr nocapture noundef %arr, i64 noundef %n) local_unnamed_addr #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata ptr %arr, metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i64 %n, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !17
  br label %for.body, !dbg !18

for.body:                                         ; preds = %entry, %for.body
; CHECK: %l1.07 = l1
  %l1.07 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  call void @llvm.dbg.value(metadata i64 %l1.07, metadata !16, metadata !DIExpression()), !dbg !17
  %add = sub nsw i64 %l1.07, 1, !dbg !20
; CHECK: %arrayidx = &arr[(l1 - 1)]
  %arrayidx = getelementptr inbounds i64, ptr %arr, i64 %add, !dbg !22
; CHECK: %0 = arr[(l1 - 1)]
  %0 = load i64, ptr %arrayidx, align 8, !dbg !22, !tbaa !23
; CHECK: %add1 = (arr[(l1 - 1)] + 10)
  %add1 = add nsw i64 %0, 10, !dbg !27
; CHECK: %arrayidx2 = &arr[l1]
  %arrayidx2 = getelementptr inbounds i64, ptr %arr, i64 %l1.07, !dbg !28
  store i64 %add1, ptr %arrayidx2, align 8, !dbg !29, !tbaa !23
  %inc = add nuw nsw i64 %l1.07, 1, !dbg !30
  call void @llvm.dbg.value(metadata i64 %inc, metadata !16, metadata !DIExpression()), !dbg !17
  %exitcond.not = icmp eq i64 %inc, 1024, !dbg !31
  br i1 %exitcond.not, label %for.end, label %for.body, !dbg !18, !llvm.loop !32

for.end:                                          ; preds = %for.body
  ret void, !dbg !36
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "approx-func-fp-math"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "loopopt-pipeline"="light" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang based Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.x.0.YYYYMMDD)", isOptimized: true, flags: " --intel -O1 -S -g -emit-llvm tm2.c -fveclib=SVML -fheinous-gnu-extensions", runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "tm2.c", directory: "/iusers/sguggill/work")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!7 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.x.0.YYYYMMDD)"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !12}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "arr", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocalVariable(name: "n", arg: 2, scope: !8, file: !1, line: 1, type: !12)
!16 = !DILocalVariable(name: "l1", scope: !8, file: !1, line: 3, type: !12)
!17 = !DILocation(line: 0, scope: !8)
!18 = !DILocation(line: 5, column: 3, scope: !19)
!19 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 3)
!20 = !DILocation(line: 6, column: 22, scope: !21)
!21 = distinct !DILexicalBlock(scope: !19, file: !1, line: 5, column: 3)
!22 = !DILocation(line: 6, column: 15, scope: !21)
!23 = !{!24, !24, i64 0}
!24 = !{!"long", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 6, column: 27, scope: !21)
!28 = !DILocation(line: 6, column: 5, scope: !21)
!29 = !DILocation(line: 6, column: 13, scope: !21)
!30 = !DILocation(line: 5, column: 29, scope: !21)
!31 = !DILocation(line: 5, column: 19, scope: !21)
!32 = distinct !{!32, !18, !33}
!33 = !DILocation(line: 6, column: 29, scope: !19)
!34 = !{!"llvm.loop.mustprogress"}
!35 = !{!"llvm.loop.unroll.disable"}
!36 = !DILocation(line: 7, column: 1, scope: !8)
