; RUN: llc -march=hexagon -O3 -o /dev/null < %s 2>&1
; Test that the compiler doesn't seg fault due to DBG_LABEL or DBG_VALUE
; in Machine Unroller pass.

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nounwind
define dso_local void @fw_time_wait_us(i32 %us) local_unnamed_addr #2 !dbg !13 {
entry:
  %dummy = alloca [256 x i8], align 8
  call void @llvm.dbg.value(metadata i32 %us, metadata !18, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %us, metadata !19, metadata !DIExpression(DW_OP_constu, 1000, DW_OP_mul, DW_OP_stack_value)), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !27
  %0 = getelementptr inbounds [256 x i8], [256 x i8]* %dummy, i32 0, i32 0, !dbg !28
  call void @llvm.lifetime.start.p0i8(i64 256, i8* nonnull %0) #3, !dbg !28
  call void @llvm.dbg.declare(metadata [256 x i8]* %dummy, metadata !22, metadata !DIExpression()), !dbg !29
  %1 = tail call i32 asm sideeffect "$0 = pcyclelo", "=r"() #3, !dbg !30, !srcloc !31
  call void @llvm.dbg.value(metadata i32 %1, metadata !21, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %us, metadata !19, metadata !DIExpression(DW_OP_constu, 1000, DW_OP_mul, DW_OP_stack_value)), !dbg !27
  %cmp10 = icmp sgt i32 %us, 0, !dbg !32
  br i1 %cmp10, label %do.body.preheader.preheader, label %while.end, !dbg !34

do.body.preheader.preheader:                      ; preds = %entry
  %mul = mul nsw i32 %us, 1000, !dbg !35
  call void @llvm.dbg.value(metadata i32 %mul, metadata !19, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %mul, metadata !19, metadata !DIExpression()), !dbg !27
  br label %do.body.preheader, !dbg !36

do.body.preheader:                                ; preds = %do.end, %do.body.preheader.preheader
  %count.011 = phi i32 [ %dec, %do.end ], [ %mul, %do.body.preheader.preheader ]
  call void @llvm.dbg.value(metadata i32 %count.011, metadata !19, metadata !DIExpression()), !dbg !27
  br label %do.body, !dbg !37

do.body:                                          ; preds = %do.body, %do.body.preheader
  %2 = tail call i32 asm sideeffect "$0 = pcyclelo", "=r"() #3, !dbg !39, !srcloc !41
  call void @llvm.dbg.value(metadata i32 %2, metadata !20, metadata !DIExpression()), !dbg !27
  %and = and i32 %2, 256, !dbg !42
  %arrayidx = getelementptr inbounds [256 x i8], [256 x i8]* %dummy, i32 0, i32 %and, !dbg !43
  %3 = load i8, i8* %arrayidx, align 8, !dbg !44, !tbaa !45
  %inc = add i8 %3, 1, !dbg !44
  store i8 %inc, i8* %arrayidx, align 8, !dbg !44, !tbaa !45
  %div = lshr i32 %2, 8, !dbg !48
  %rem = and i32 %div, 255, !dbg !49
  %arrayidx1 = getelementptr inbounds [256 x i8], [256 x i8]* %dummy, i32 0, i32 %rem, !dbg !50
  %4 = load i8, i8* %arrayidx1, align 1, !dbg !51, !tbaa !45
  %inc2 = add i8 %4, 1, !dbg !51
  store i8 %inc2, i8* %arrayidx1, align 1, !dbg !51, !tbaa !45
  %cmp3 = icmp ugt i32 %2, %1, !dbg !52
  br i1 %cmp3, label %do.end, label %do.body, !dbg !53, !llvm.loop !55

do.end:                                           ; preds = %do.body
  %dec = add nsw i32 %count.011, -1, !dbg !57
  call void @llvm.dbg.value(metadata i32 %dec, metadata !19, metadata !DIExpression()), !dbg !27
  %cmp = icmp sgt i32 %dec, 0, !dbg !32
  br i1 %cmp, label %do.body.preheader, label %while.end, !dbg !34, !llvm.loop !58

while.end:                                        ; preds = %do.end, %entry
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %0) #3, !dbg !60
  ret void, !dbg !60
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { argmemonly nounwind willreturn }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv71" "target-features"="+v71,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "QuIC LLVM Hexagon Clang version 8.4.alpha4 Engineering Release: hexagon-clang-mono-84-2032 (based on LLVM 10.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "/prj/dsp/qdsp6/austin/builds/hexbuild/test_trees/MASTER/test/regress/simulator/api_tests/NMI/test.c", directory: "/local/mnt/workspace")
!2 = !{}
!3 = !{!4, !6, !8}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 32)
!5 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"QuIC LLVM Hexagon Clang version 8.4.alpha4 Engineering Release: hexagon-clang-mono-84-2032 (based on LLVM 10.0.0)"}
!13 = distinct !DISubprogram(name: "fw_time_wait_us", scope: !14, file: !14, line: 71, type: !15, scopeLine: 72, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!14 = !DIFile(filename: "/prj/dsp/qdsp6/austin/builds/hexbuild/test_trees/MASTER/test/regress/simulator/api_tests/NMI/test.c", directory: "")
!15 = !DISubroutineType(types: !16)
!16 = !{null, !7}
!17 = !{!18, !19, !20, !21, !22}
!18 = !DILocalVariable(name: "us", arg: 1, scope: !13, file: !14, line: 71, type: !7)
!19 = !DILocalVariable(name: "count", scope: !13, file: !14, line: 73, type: !7)
!20 = !DILocalVariable(name: "xo", scope: !13, file: !14, line: 74, type: !5)
!21 = !DILocalVariable(name: "base", scope: !13, file: !14, line: 74, type: !5)
!22 = !DILocalVariable(name: "dummy", scope: !13, file: !14, line: 75, type: !23)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 2048, elements: !25)
!24 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!25 = !{!26}
!26 = !DISubrange(count: 256)
!27 = !DILocation(line: 0, scope: !13)
!28 = !DILocation(line: 75, column: 3, scope: !13)
!29 = !DILocation(line: 75, column: 17, scope: !13)
!30 = !DILocation(line: 77, column: 3, scope: !13)
!31 = !{i32 1828}
!32 = !DILocation(line: 78, column: 16, scope: !33)
!33 = !DILexicalBlockFile(scope: !13, file: !14, discriminator: 2)
!34 = !DILocation(line: 78, column: 3, scope: !33)
!35 = !DILocation(line: 73, column: 20, scope: !13)
!36 = !DILocation(line: 78, column: 3, scope: !13)
!37 = !DILocation(line: 80, column: 5, scope: !38)
!38 = distinct !DILexicalBlock(scope: !13, file: !14, line: 79, column: 3)
!39 = !DILocation(line: 82, column: 7, scope: !40)
!40 = distinct !DILexicalBlock(scope: !38, file: !14, line: 81, column: 5)
!41 = !{i32 1916}
!42 = !DILocation(line: 84, column: 17, scope: !40)
!43 = !DILocation(line: 84, column: 7, scope: !40)
!44 = !DILocation(line: 84, column: 24, scope: !40)
!45 = !{!46, !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C/C++ TBAA"}
!48 = !DILocation(line: 85, column: 17, scope: !40)
!49 = !DILocation(line: 85, column: 23, scope: !40)
!50 = !DILocation(line: 85, column: 7, scope: !40)
!51 = !DILocation(line: 85, column: 29, scope: !40)
!52 = !DILocation(line: 87, column: 15, scope: !38)
!53 = !DILocation(line: 86, column: 5, scope: !54)
!54 = !DILexicalBlockFile(scope: !40, file: !14, discriminator: 2)
!55 = distinct !{!55, !37, !56}
!56 = !DILocation(line: 87, column: 22, scope: !38)
!57 = !DILocation(line: 88, column: 10, scope: !38)
!58 = distinct !{!58, !36, !59}
!59 = !DILocation(line: 89, column: 3, scope: !13)
!60 = !DILocation(line: 90, column: 1, scope: !13)
