; RUN: opt < %s -passes=loop-idiom -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local void @fun(ptr noundef %a) #0 !dbg !10 {

; CHECK-LABEL: entry:
; CHECK-NOT: call void @llvm.memset.p0.i64{{.*}}dbg {{![0-9]+}}
entry:
  tail call void @llvm.dbg.value(metadata ptr %a, metadata !16, metadata !DIExpression()), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !21
  br label %for.body, !dbg !22

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %i.01, metadata !18, metadata !DIExpression()), !dbg !21
  %idxprom = sext i32 %i.01 to i64, !dbg !23
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %idxprom, !dbg !23
  store double 0.000000e+00, ptr %arrayidx, align 8, !dbg !26
  %inc = add nsw i32 %i.01, 1, !dbg !27
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !18, metadata !DIExpression()), !dbg !21
  %cmp = icmp slt i32 %inc, 1000, !dbg !28
  br i1 %cmp, label %for.body, label %for.end, !dbg !22, !llvm.loop !29

for.end:                                          ; preds = %for.body
  ret void, !dbg !32
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 !dbg !33 {
entry:
  %a = alloca [1000 x double], align 16
  call void @llvm.dbg.declare(metadata ptr %a, metadata !36, metadata !DIExpression()), !dbg !40
  %arraydecay = getelementptr inbounds [1000 x double], ptr %a, i64 0, i64 0, !dbg !41
  call void @fun(ptr noundef %arraydecay), !dbg !42
  ret i32 0, !dbg !43
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0git (https://github.com/llvm/llvm-project.git 7e604485e18d40be6ce6310e4a3e583ca0b7df47)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "1122.c", directory: "/home/linuxbrew/llvm-debug/LoopIdiomRecognize", checksumkind: CSK_MD5, checksum: "d9d161f9aca9398de6de8509e8cd8335")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!15 = !{}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 2, type: !20)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 5)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 0, scope: !19)
!22 = !DILocation(line: 2, column: 5, scope: !19)
!23 = !DILocation(line: 3, column: 9, scope: !24)
!24 = distinct !DILexicalBlock(scope: !25, file: !1, line: 2, column: 36)
!25 = distinct !DILexicalBlock(scope: !19, file: !1, line: 2, column: 5)
!26 = !DILocation(line: 3, column: 14, scope: !24)
!27 = !DILocation(line: 2, column: 32, scope: !25)
!28 = !DILocation(line: 2, column: 23, scope: !25)
!29 = distinct !{!29, !22, !30, !31}
!30 = !DILocation(line: 4, column: 5, scope: !19)
!31 = !{!"llvm.loop.mustprogress"}
!32 = !DILocation(line: 5, column: 1, scope: !10)
!33 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !34, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!34 = !DISubroutineType(types: !35)
!35 = !{!20}
!36 = !DILocalVariable(name: "a", scope: !33, file: !1, line: 8, type: !37)
!37 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 64000, elements: !38)
!38 = !{!39}
!39 = !DISubrange(count: 1000)
!40 = !DILocation(line: 8, column: 12, scope: !33)
!41 = !DILocation(line: 9, column: 9, scope: !33)
!42 = !DILocation(line: 9, column: 5, scope: !33)
!43 = !DILocation(line: 10, column: 5, scope: !33)
