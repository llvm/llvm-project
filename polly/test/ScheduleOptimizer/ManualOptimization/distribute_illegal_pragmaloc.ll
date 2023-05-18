; RUN: opt %loadPolly -polly-opt-isl -polly-reschedule=0 -polly-pragma-based-opts=1 -disable-output < %s 2>&1 | FileCheck %s --match-full-lines
;
; CHECK: warning: distribute_illegal.c:1:42: not applying loop fission/distribution: cannot ensure semantic equivalence due to possible dependency violations
;
; void foo(double *A,double *B) {
;   for (int i = 1; i < 128; ++i) {
;     A[i] = i;
;     B[i] = A[i+1];
;   }
; }

source_filename = "distribute_illegal.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo(ptr %A, ptr %B) #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata ptr %A, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata ptr %B, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !19
  br label %for.cond, !dbg !20

for.cond:
  %i.0 = phi i32 [ 1, %entry ], [ %inc, %for.body ], !dbg !19
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !15, metadata !DIExpression()), !dbg !19
  %cmp = icmp slt i32 %i.0, 128, !dbg !21
  br i1 %cmp, label %for.body, label %for.end, !dbg !23

for.body:
  %conv = sitofp i32 %i.0 to double, !dbg !24
  %idxprom = sext i32 %i.0 to i64, !dbg !26
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom, !dbg !26
  store double %conv, ptr %arrayidx, align 8, !dbg !27, !tbaa !28

  %add = add nsw i32 %i.0, 1, !dbg !32
  %idxprom1 = sext i32 %add to i64, !dbg !33
  %arrayidx2 = getelementptr inbounds double, ptr %A, i64 %idxprom1, !dbg !33
  %0 = load double, ptr %arrayidx2, align 8, !dbg !33, !tbaa !28
  %idxprom3 = sext i32 %i.0 to i64, !dbg !34
  %arrayidx4 = getelementptr inbounds double, ptr %B, i64 %idxprom3, !dbg !34
  store double %0, ptr %arrayidx4, align 8, !dbg !35, !tbaa !28

  %inc = add nsw i32 %i.0, 1, !dbg !36
  call void @llvm.dbg.value(metadata i32 %inc, metadata !15, metadata !DIExpression()), !dbg !19
  br label %for.cond, !dbg !37, !llvm.loop !38

for.end:
  ret void, !dbg !41
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (/home/meinersbur/src/llvm-project/clang 81189783049d2b93f653c121d3731fd1732a3916)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "distribute_illegal.c", directory: "/path/to")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0 (/home/meinersbur/src/llvm-project/clang 81189783049d2b93f653c121d3731fd1732a3916)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "B", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "i", scope: !16, file: !1, line: 2, type: !17)
!16 = distinct !DILexicalBlock(scope: !7, file: !1, line: 2, column: 3)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 0, scope: !16)
!20 = !DILocation(line: 2, column: 8, scope: !16)
!21 = !DILocation(line: 2, column: 21, scope: !22)
!22 = distinct !DILexicalBlock(scope: !16, file: !1, line: 2, column: 3)
!23 = !DILocation(line: 2, column: 3, scope: !16)
!24 = !DILocation(line: 3, column: 12, scope: !25)
!25 = distinct !DILexicalBlock(scope: !22, file: !1, line: 2, column: 33)
!26 = !DILocation(line: 3, column: 5, scope: !25)
!27 = !DILocation(line: 3, column: 10, scope: !25)
!28 = !{!29, !29, i64 0}
!29 = !{!"double", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !DILocation(line: 4, column: 15, scope: !25)
!33 = !DILocation(line: 4, column: 12, scope: !25)
!34 = !DILocation(line: 4, column: 5, scope: !25)
!35 = !DILocation(line: 4, column: 10, scope: !25)
!36 = !DILocation(line: 2, column: 28, scope: !22)
!37 = !DILocation(line: 2, column: 3, scope: !22)
!38 = distinct !{!38, !23, !39, !40, !100, !101}
!39 = !DILocation(line: 5, column: 3, scope: !16)
!40 = !{!"llvm.loop.mustprogress"}
!41 = !DILocation(line: 6, column: 1, scope: !7)
!100 = !{!"llvm.loop.distribute.enable"}
!101 = !{!"llvm.loop.distribute.loc", !102}
!102 = !DILocation(line: 1, column: 42, scope: !16)
