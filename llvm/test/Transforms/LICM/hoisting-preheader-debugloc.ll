; RUN: opt -passes=licm %s -S | FileCheck %s

; CHECK: %arrayidx4.promoted = load i32, ptr %arrayidx4, align 4
; CHECK-NOT: !dbg
; CHECK: br label %for.body

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@a = dso_local local_unnamed_addr global i16 0, align 2, !dbg !0
@b = dso_local local_unnamed_addr global i16 0, align 2, !dbg !6

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare i16 @e(i32)

define i16 @g() !dbg !13 {
entry:
  %l_284 = alloca [2 x [3 x [6 x i32]]], align 16
  call void @llvm.lifetime.start.p0(i64 144, ptr nonnull %l_284), !dbg !24
  call void @llvm.dbg.declare(metadata ptr %l_284, metadata !17, metadata !DIExpression()), !dbg !25
  %0 = load i16, ptr @a, align 2, !dbg !26, !tbaa !29
  %cmp11 = icmp sgt i16 %0, -1, !dbg !33
  br i1 %cmp11, label %for.body.lr.ph, label %cleanup, !dbg !34

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !34

for.body:                                         ; preds = %for.cond2, %for.body.lr.ph
  %call = call signext i16 @e(i32 1), !dbg !35
  %tobool = icmp eq i16 %call, 0, !dbg !35
  br i1 %tobool, label %for.cond2, label %for.body.cleanup_crit_edge, !dbg !38

for.cond2:                                        ; preds = %for.body
  %arrayidx4 = getelementptr inbounds [2 x [3 x [6 x i32]]], ptr %l_284, i64 0, i64 1, i64 2, i64 5, !dbg !39
  %l = load i32, ptr %arrayidx4, !dbg !43, !tbaa !44
  %add = add i32 %l, 1, !dbg !43
  store i32 %add, ptr %arrayidx4, align 4, !dbg !43, !tbaa !44
  %arrayidx8 = getelementptr inbounds [2 x [3 x [6 x i32]]], ptr %l_284, i64 0, i64 1, i64 1, i64 4, !dbg !46
  %1 = load i32, ptr %arrayidx8, align 8, !dbg !46, !tbaa !44
  %conv9 = trunc i32 %1 to i16, !dbg !46
  store i16 %conv9, ptr @b, align 2, !dbg !47, !tbaa !29
  %2 = load i16, ptr @a, align 2, !dbg !26, !tbaa !29
  %cmp = icmp sgt i16 %2, -1, !dbg !33
  br i1 %cmp, label %for.body, label %for.cond.cleanup_crit_edge, !dbg !34, !llvm.loop !48

for.cond.cleanup_crit_edge:                       ; preds = %for.cond2
  br label %cleanup, !dbg !34

for.body.cleanup_crit_edge:                       ; preds = %for.body
  br label %cleanup, !dbg !38

cleanup:                                          ; preds = %for.body.cleanup_crit_edge, %for.cond.cleanup_crit_edge, %entry
  call void @llvm.lifetime.end.p0(i64 144, ptr nonnull %l_284), !dbg !51
  ret i16 1, !dbg !51
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 237d0e3c0416abf9919406bcc92874cfd15f5e0c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "a.c", directory: "/home/davide/lldb-build/bin")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 237d0e3c0416abf9919406bcc92874cfd15f5e0c)"}
!13 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 5, type: !14, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!8}
!16 = !{!17}
!17 = !DILocalVariable(name: "l_284", scope: !13, file: !3, line: 6, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !19, size: 1152, elements: !20)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !{!21, !22, !23}
!21 = !DISubrange(count: 2)
!22 = !DISubrange(count: 3)
!23 = !DISubrange(count: 6)
!24 = !DILocation(line: 6, column: 3, scope: !13)
!25 = !DILocation(line: 6, column: 7, scope: !13)
!26 = !DILocation(line: 7, column: 10, scope: !27)
!27 = distinct !DILexicalBlock(scope: !28, file: !3, line: 7, column: 3)
!28 = distinct !DILexicalBlock(scope: !13, file: !3, line: 7, column: 3)
!29 = !{!30, !30, i64 0}
!30 = !{!"short", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 7, column: 12, scope: !27)
!34 = !DILocation(line: 7, column: 3, scope: !28)
!35 = !DILocation(line: 8, column: 9, scope: !36)
!36 = distinct !DILexicalBlock(scope: !37, file: !3, line: 8, column: 9)
!37 = distinct !DILexicalBlock(scope: !27, file: !3, line: 7, column: 19)
!38 = !DILocation(line: 8, column: 9, scope: !37)
!39 = !DILocation(line: 11, column: 7, scope: !40)
!40 = distinct !DILexicalBlock(scope: !41, file: !3, line: 10, column: 14)
!41 = distinct !DILexicalBlock(scope: !42, file: !3, line: 10, column: 5)
!42 = distinct !DILexicalBlock(scope: !37, file: !3, line: 10, column: 5)
!43 = !DILocation(line: 11, column: 22, scope: !40)
!44 = !{!45, !45, i64 0}
!45 = !{!"int", !31, i64 0}
!46 = !DILocation(line: 16, column: 9, scope: !37)
!47 = !DILocation(line: 16, column: 7, scope: !37)
!48 = distinct !{!48, !34, !49, !50}
!49 = !DILocation(line: 17, column: 3, scope: !28)
!50 = !{!"llvm.loop.unroll.disable"}
!51 = !DILocation(line: 18, column: 1, scope: !13)
