target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Function Attrs: nounwind readnone uwtable willreturn
define dso_local i32 @bar(i32 %x, i32 %y) local_unnamed_addr #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %y, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !18
  %rem = srem i32 %x, 3, !dbg !20
  %tobool.not = icmp eq i32 %rem, 0, !dbg !20
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 2, i32 2, i64 -1), !dbg !21
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 3, i32 2, i64 -1), !dbg !23
  %0 = sub i32 0, %y, !dbg !24
  %retval.0.p = select i1 %tobool.not, i32 %y, i32 %0, !dbg !24
  %retval.0 = add i32 %retval.0.p, %x, !dbg !24
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 4, i32 0, i64 -1), !dbg !25
  ret i32 %retval.0, !dbg !25
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() local_unnamed_addr #1 !dbg !26 {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !32
  call void @llvm.dbg.value(metadata i32 0, metadata !30, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 0, metadata !31, metadata !DIExpression()), !dbg !33
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !34
  call void @llvm.dbg.value(metadata i32 1, metadata !31, metadata !DIExpression()), !dbg !33
  br label %while.body, !dbg !35

while.body:                                       ; preds = %entry, %if.end
  %inc8 = phi i32 [ 1, %entry ], [ %inc, %if.end ]
  %s.07 = phi i32 [ 0, %entry ], [ %s.1, %if.end ]
  call void @llvm.dbg.value(metadata i32 %s.07, metadata !30, metadata !DIExpression()), !dbg !33
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 3, i32 0, i64 -1), !dbg !36
  %rem = urem i32 %inc8, 91, !dbg !38
  %tobool.not = icmp eq i32 %rem, 0, !dbg !38
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !39

if.then:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !40
  call void @llvm.dbg.value(metadata i32 %inc8, metadata !15, metadata !DIExpression()) #6, !dbg !41
  call void @llvm.dbg.value(metadata i32 %s.07, metadata !16, metadata !DIExpression()) #6, !dbg !41
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1) #6, !dbg !44
  %rem.i = urem i32 %inc8, 3, !dbg !45
  %tobool.not.i = icmp eq i32 %rem.i, 0, !dbg !45
  %0 = sub i32 0, %s.07, !dbg !48
  %retval.0.p.i = select i1 %tobool.not.i, i32 %s.07, i32 %0, !dbg !48
  %retval.0.i = add i32 %retval.0.p.i, %inc8, !dbg !48
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 4, i32 0, i64 -1) #6, !dbg !49
  call void @llvm.dbg.value(metadata i32 %retval.0.i, metadata !30, metadata !DIExpression()), !dbg !33
  br label %if.end, !dbg !50

if.else:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1), !dbg !51
  %add = add nsw i32 %s.07, 30, !dbg !51
  call void @llvm.dbg.value(metadata i32 %add, metadata !30, metadata !DIExpression()), !dbg !33
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %s.1 = phi i32 [ %retval.0.i, %if.then ], [ %add, %if.else ], !dbg !52
  call void @llvm.dbg.value(metadata i32 %s.1, metadata !30, metadata !DIExpression()), !dbg !33
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1), !dbg !35
  call void @llvm.dbg.value(metadata i32 %inc8, metadata !31, metadata !DIExpression()), !dbg !33
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !34
  %inc = add nuw nsw i32 %inc8, 1, !dbg !34
  call void @llvm.dbg.value(metadata i32 %inc, metadata !31, metadata !DIExpression()), !dbg !33
  %exitcond.not = icmp eq i32 %inc, 16000001, !dbg !53
  br i1 %exitcond.not, label %while.end, label %while.body, !dbg !35, !llvm.loop !54

while.end:                                        ; preds = %if.end
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !57
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i32 %s.1), !dbg !58
  ret void, !dbg !60
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #3 !dbg !61 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !64
  call void @foo(), !dbg !65
  ret i32 0, !dbg !67
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nounwind readnone uwtable willreturn "disable-tail-calls"="true" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nounwind "disable-tail-calls"="true" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { inaccessiblememonly nounwind willreturn }
attributes #5 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}
!llvm.pseudo_probe_desc = !{!7, !8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "test")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = !{i64 -2012135647395072713, i64 72617220756, !"bar", null}
!8 = !{i64 6699318081062747564, i64 563088904013236, !"foo", null}
!9 = !{i64 -2624081020897602054, i64 281479271677951, !"main", null}
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 3, type: !13)
!16 = !DILocalVariable(name: "y", arg: 2, scope: !10, file: !1, line: 3, type: !13)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 4, column: 9, scope: !19)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 4, column: 9)
!20 = !DILocation(line: 4, column: 11, scope: !19)
!21 = !DILocation(line: 5, column: 18, scope: !22)
!22 = distinct !DILexicalBlock(scope: !19, file: !1, line: 4, column: 16)
!23 = !DILocation(line: 7, column: 14, scope: !10)
!24 = !DILocation(line: 4, column: 9, scope: !10)
!25 = !DILocation(line: 8, column: 1, scope: !10)
!26 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 11, type: !27, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!27 = !DISubroutineType(types: !28)
!28 = !{null}
!29 = !{!30, !31}
!30 = !DILocalVariable(name: "s", scope: !26, file: !1, line: 12, type: !13)
!31 = !DILocalVariable(name: "i", scope: !26, file: !1, line: 12, type: !13)
!32 = !DILocation(line: 12, column: 5, scope: !26)
!33 = !DILocation(line: 0, scope: !26)
!34 = !DILocation(line: 13, column: 15, scope: !26)
!35 = !DILocation(line: 13, column: 7, scope: !26)
!36 = !DILocation(line: 14, column: 17, scope: !37)
!37 = distinct !DILexicalBlock(scope: !26, file: !1, line: 14, column: 17)
!38 = !DILocation(line: 14, column: 19, scope: !37)
!39 = !DILocation(line: 14, column: 17, scope: !26)
!40 = !DILocation(line: 14, column: 33, scope: !37)
!41 = !DILocation(line: 0, scope: !10, inlinedAt: !42)
!42 = distinct !DILocation(line: 14, column: 29, scope: !43)
!43 = !DILexicalBlockFile(scope: !37, file: !1, discriminator: 186646599)
!44 = !DILocation(line: 4, column: 9, scope: !19, inlinedAt: !42)
!45 = !DILocation(line: 4, column: 11, scope: !19, inlinedAt: !42)
!46 = !DILocation(line: 5, column: 18, scope: !22, inlinedAt: !42)
!47 = !DILocation(line: 7, column: 14, scope: !10, inlinedAt: !42)
!48 = !DILocation(line: 4, column: 9, scope: !10, inlinedAt: !42)
!49 = !DILocation(line: 8, column: 1, scope: !10, inlinedAt: !42)
!50 = !DILocation(line: 14, column: 25, scope: !37)
!51 = !DILocation(line: 14, column: 47, scope: !37)
!52 = !DILocation(line: 0, scope: !37)
!53 = !DILocation(line: 13, column: 18, scope: !26)
!54 = distinct !{!54, !35, !55, !56}
!55 = !DILocation(line: 14, column: 50, scope: !26)
!56 = !{!"llvm.loop.mustprogress"}
!57 = !DILocation(line: 15, column: 31, scope: !26)
!58 = !DILocation(line: 15, column: 9, scope: !59)
!59 = !DILexicalBlockFile(scope: !26, file: !1, discriminator: 186646607)
!60 = !DILocation(line: 16, column: 1, scope: !26)
!61 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 18, type: !62, scopeLine: 18, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!62 = !DISubroutineType(types: !63)
!63 = !{!13}
!64 = !DILocation(line: 19, column: 5, scope: !61)
!65 = !DILocation(line: 19, column: 5, scope: !66)
!66 = !DILexicalBlockFile(scope: !61, file: !1, discriminator: 7)
!67 = !DILocation(line: 20, column: 7, scope: !61)
