; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=simplifycfg -S -o %t
; RUN: FileCheck %s < %t

; Test to make sure the dangling probe is gone.
; CHECK: define dso_local i32 @foo
; CHECK-NOT: call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 5

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(ptr nocapture %marker, ptr nocapture %move_ordering, ptr nocapture %moves, i32 %num_moves) local_unnamed_addr #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata ptr %marker, metadata !19, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata ptr %move_ordering, metadata !20, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata ptr %moves, metadata !21, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %num_moves, metadata !22, metadata !DIExpression()), !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 1, i32 0, i64 -1), !dbg !29
  call void @llvm.dbg.value(metadata i32 -1000000, metadata !25, metadata !DIExpression()), !dbg !28
  %0 = load i32, ptr %marker, align 4, !dbg !30, !tbaa !31
  %inc = add nsw i32 %0, 1, !dbg !30
  store i32 %inc, ptr %marker, align 4, !dbg !30, !tbaa !31
  call void @llvm.dbg.value(metadata i32 %inc, metadata !27, metadata !DIExpression()), !dbg !28
  %cmp = icmp slt i32 %0, 9, !dbg !35
  br i1 %cmp, label %for.cond, label %if.else, !dbg !37

for.cond:                                         ; preds = %entry, %for.inc
  %i.0 = phi i32 [ %inc6, %for.inc ], [ %inc, %entry ], !dbg !38
  %best.0 = phi i32 [ %best.1, %for.inc ], [ -1000000, %entry ], !dbg !28
  %tmp.0 = phi i32 [ %tmp.1, %for.inc ], [ undef, %entry ]
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 2, i32 2, i64 -1), !dbg !41
  call void @llvm.dbg.value(metadata i32 %tmp.0, metadata !26, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %best.0, metadata !25, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !24, metadata !DIExpression()), !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 3, i32 0, i64 -1), !dbg !42
  %cmp1 = icmp slt i32 %i.0, %num_moves, !dbg !44
  br i1 %cmp1, label %for.body, label %if.end12, !dbg !45

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 4, i32 0, i64 -1), !dbg !46
  %idxprom = sext i32 %i.0 to i64, !dbg !46
  %arrayidx = getelementptr inbounds i32, ptr %move_ordering, i64 %idxprom, !dbg !46
  %1 = load i32, ptr %arrayidx, align 4, !dbg !46, !tbaa !31
  %cmp2 = icmp sgt i32 %1, %best.0, !dbg !49
  br i1 %cmp2, label %if.then3, label %for.inc, !dbg !50

if.then3:                                         ; preds = %for.body
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 5, i32 0, i64 -1), !dbg !51
  call void @llvm.dbg.value(metadata i32 %1, metadata !25, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !26, metadata !DIExpression()), !dbg !28
  br label %for.inc, !dbg !53

for.inc:                                          ; preds = %for.body, %if.then3
  %best.1 = phi i32 [ %1, %if.then3 ], [ %best.0, %for.body ], !dbg !28
  %tmp.1 = phi i32 [ %i.0, %if.then3 ], [ %tmp.0, %for.body ]
  call void @llvm.dbg.value(metadata i32 %tmp.1, metadata !26, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %best.1, metadata !25, metadata !DIExpression()), !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 6, i32 0, i64 -1), !dbg !54
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 7, i32 0, i64 -1), !dbg !55
  %inc6 = add nsw i32 %i.0, 1, !dbg !55
  call void @llvm.dbg.value(metadata i32 %inc6, metadata !24, metadata !DIExpression()), !dbg !28
  br label %for.cond, !dbg !56, !llvm.loop !57

if.else:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 9, i32 0, i64 -1), !dbg !59
  %cmp7 = icmp slt i32 %inc, %num_moves, !dbg !61
  br i1 %cmp7, label %cleanup, label %if.end12, !dbg !62

if.end12:                                         ; preds = %if.else, %for.cond
  %best.2 = phi i32 [ %best.0, %for.cond ], [ -1000000, %if.else ], !dbg !63
  %tmp.2 = phi i32 [ %tmp.0, %for.cond ], [ undef, %if.else ]
  call void @llvm.dbg.value(metadata i32 %tmp.2, metadata !26, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %best.2, metadata !25, metadata !DIExpression()), !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 11, i32 2, i64 -1), !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 8, i32 2, i64 -1), !dbg !64
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 12, i32 0, i64 -1), !dbg !65
  %cmp13 = icmp sgt i32 %best.2, -1000000, !dbg !67
  br i1 %cmp13, label %if.then14, label %cleanup, !dbg !68

if.then14:                                        ; preds = %if.end12
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 13, i32 0, i64 -1), !dbg !69
  br label %cleanup, !dbg !78

cleanup:                                          ; preds = %if.end12, %if.else, %if.then14
  %retval.0 = phi i32 [ 1, %if.then14 ], [ 1, %if.else ], [ 0, %if.end12 ], !dbg !28
  call void @llvm.pseudoprobe(i64 -4224472938262609671, i64 14, i32 2, i64 -1), !dbg !79
  ret i32 %retval.0, !dbg !83
}

; Function Attrs: inaccessiblememonly nounwind willreturn mustprogress
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { inaccessiblememonly nounwind willreturn mustprogress }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!llvm.pseudo_probe_desc = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"ThinLTO", i32 0}
!7 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!8 = !{!"clang version 13.0.0"}
!9 = !{i64 -4224472938262609671, i64 328037311046, !"foo", null}
!10 = distinct !DISubprogram(name: "remove_one_fast", linkageName: "foo", scope: !1, file: !1, line: 8, type: !11, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14, !14, !15, !17}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "move_s", file: !1, line: 6, baseType: !13)
!17 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!18 = !{!19, !20, !21, !22, !23, !24, !25, !26, !27}
!19 = !DILocalVariable(name: "marker", arg: 1, scope: !10, file: !1, line: 8, type: !14)
!20 = !DILocalVariable(name: "move_ordering", arg: 2, scope: !10, file: !1, line: 8, type: !14)
!21 = !DILocalVariable(name: "moves", arg: 3, scope: !10, file: !1, line: 9, type: !15)
!22 = !DILocalVariable(name: "num_moves", arg: 4, scope: !10, file: !1, line: 9, type: !17)
!23 = !DILocalVariable(name: "tmpmv", scope: !10, file: !1, line: 10, type: !16)
!24 = !DILocalVariable(name: "i", scope: !10, file: !1, line: 11, type: !13)
!25 = !DILocalVariable(name: "best", scope: !10, file: !1, line: 11, type: !13)
!26 = !DILocalVariable(name: "tmp", scope: !10, file: !1, line: 12, type: !13)
!27 = !DILocalVariable(name: "mark", scope: !10, file: !1, line: 12, type: !13)
!28 = !DILocation(line: 0, scope: !10)
!29 = !DILocation(line: 10, column: 5, scope: !10)
!30 = !DILocation(line: 14, column: 14, scope: !10)
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C++ TBAA"}
!35 = !DILocation(line: 17, column: 14, scope: !36)
!36 = distinct !DILexicalBlock(scope: !10, file: !1, line: 17, column: 9)
!37 = !DILocation(line: 17, column: 9, scope: !10)
!38 = !DILocation(line: 0, scope: !39)
!39 = distinct !DILexicalBlock(scope: !40, file: !1, line: 18, column: 9)
!40 = distinct !DILexicalBlock(scope: !36, file: !1, line: 17, column: 20)
!41 = !DILocation(line: 18, column: 18, scope: !39)
!42 = !DILocation(line: 18, column: 24, scope: !43)
!43 = distinct !DILexicalBlock(scope: !39, file: !1, line: 18, column: 9)
!44 = !DILocation(line: 18, column: 26, scope: !43)
!45 = !DILocation(line: 18, column: 9, scope: !39)
!46 = !DILocation(line: 19, column: 17, scope: !47)
!47 = distinct !DILexicalBlock(scope: !48, file: !1, line: 19, column: 17)
!48 = distinct !DILexicalBlock(scope: !43, file: !1, line: 18, column: 44)
!49 = !DILocation(line: 19, column: 34, scope: !47)
!50 = !DILocation(line: 19, column: 17, scope: !48)
!51 = !DILocation(line: 20, column: 24, scope: !52)
!52 = distinct !DILexicalBlock(scope: !47, file: !1, line: 19, column: 42)
!53 = !DILocation(line: 22, column: 13, scope: !52)
!54 = !DILocation(line: 23, column: 9, scope: !48)
!55 = !DILocation(line: 18, column: 40, scope: !43)
!56 = !DILocation(line: 18, column: 9, scope: !43)
!57 = distinct !{!57, !45, !58}
!58 = !DILocation(line: 23, column: 9, scope: !39)
!59 = !DILocation(line: 24, column: 16, scope: !60)
!60 = distinct !DILexicalBlock(scope: !36, file: !1, line: 24, column: 16)
!61 = !DILocation(line: 24, column: 21, scope: !60)
!62 = !DILocation(line: 24, column: 16, scope: !36)
!63 = !DILocation(line: 11, column: 12, scope: !10)
!64 = !DILocation(line: 24, column: 5, scope: !40)
!65 = !DILocation(line: 31, column: 9, scope: !66)
!66 = distinct !DILexicalBlock(scope: !10, file: !1, line: 31, column: 9)
!67 = !DILocation(line: 31, column: 14, scope: !66)
!68 = !DILocation(line: 31, column: 9, scope: !10)
!69 = !DILocation(line: 35, column: 30, scope: !70)
!70 = distinct !DILexicalBlock(scope: !66, file: !1, line: 31, column: 22)
!71 = !DILocation(line: 35, column: 9, scope: !70)
!72 = !DILocation(line: 35, column: 28, scope: !70)
!73 = !DILocation(line: 36, column: 29, scope: !70)
!74 = !DILocation(line: 38, column: 17, scope: !70)
!75 = !DILocation(line: 39, column: 23, scope: !70)
!76 = !DILocation(line: 39, column: 21, scope: !70)
!77 = !DILocation(line: 40, column: 20, scope: !70)
!78 = !DILocation(line: 42, column: 9, scope: !70)
!79 = !DILocation(line: 44, column: 9, scope: !80)
!80 = distinct !DILexicalBlock(scope: !66, file: !1, line: 43, column: 12)
!81 = !DILocation(line: 25, column: 16, scope: !82)
!82 = distinct !DILexicalBlock(scope: !60, file: !1, line: 24, column: 34)
!83 = !DILocation(line: 46, column: 1, scope: !10)
