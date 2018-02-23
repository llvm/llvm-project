; RUN: llc < %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.A = type { i8 }
%class.B = type { i8 }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"false\00", align 1

define void @_Z11PrintVectorv() local_unnamed_addr #0 !dbg !6 {
entry:
  %agg.tmp.i.i = alloca %class.A, align 1
  %text.i = alloca %class.A, align 1
  %v = alloca %class.B, align 1
  %0 = getelementptr inbounds %class.B, %class.B* %v, i64 0, i32 0, !dbg !40
  call void @llvm.lifetime.start(i64 1, i8* %0) #4, !dbg !40
  %1 = getelementptr inbounds %class.A, %class.A* %text.i, i64 0, i32 0, !dbg !41
  %2 = getelementptr inbounds %class.A, %class.A* %agg.tmp.i.i, i64 0, i32 0, !dbg !59
  br label %for.cond, !dbg !65

for.cond:                                         ; preds = %for.cond, %entry
  call void @llvm.dbg.value(metadata %class.B* %v, i64 0, metadata !29, metadata !66), !dbg !67
  %call = call double @_ZN1BixEj(%class.B* nonnull %v, i32 undef), !dbg !68
  call void @llvm.dbg.value(metadata double %call, i64 0, metadata !49, metadata !69), !dbg !70
  call void @llvm.dbg.value(metadata i32* null, i64 0, metadata !52, metadata !69), !dbg !71
  call void @llvm.dbg.value(metadata %class.A* undef, i64 0, metadata !54, metadata !69), !dbg !72
  call void @llvm.lifetime.start(i64 1, i8* %1) #4, !dbg !41
  %tobool.i = fcmp une double %call, 0.000000e+00, !dbg !73
  %cond.i = select i1 %tobool.i, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), !dbg !73
  call void @llvm.dbg.value(metadata %class.A* %text.i, i64 0, metadata !55, metadata !66), !dbg !74
  call void @llvm.lifetime.start(i64 1, i8* %2), !dbg !59
  call void @llvm.dbg.value(metadata %class.A* %text.i, i64 0, metadata !62, metadata !69), !dbg !59
  call void @llvm.dbg.value(metadata i8* %cond.i, i64 0, metadata !63, metadata !69), !dbg !75
  call void @_ZN1AC1EPKc(%class.A* nonnull %agg.tmp.i.i, i8* %cond.i), !dbg !76
  call void @_ZN1A5m_fn1ES_(%class.A* nonnull %text.i), !dbg !77
  call void @llvm.lifetime.end(i64 1, i8* %2), !dbg !79
  call void @llvm.lifetime.end(i64 1, i8* %1) #4, !dbg !80
  br label %for.cond, !dbg !81, !llvm.loop !82
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare double @_ZN1BixEj(%class.B*, i32) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

declare void @_ZN1A5m_fn1ES_(%class.A*) local_unnamed_addr #2

declare void @_ZN1AC1EPKc(%class.A*, i8*) unnamed_addr #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #0 = { noreturn uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273450) (llvm/trunk 273521)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/usr/local/google/home/niravd/bug_28270.c", directory: "/usr/local/google/home/niravd/build/llvm/build_debug")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 273450) (llvm/trunk 273521)"}
!6 = distinct !DISubprogram(name: "PrintVector", linkageName: "_Z11PrintVectorv", scope: !1, file: !1, line: 18, type: !7, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10, !25, !27, !28, !29, !38}
!10 = !DILocalVariable(name: "_text", scope: !6, file: !1, line: 19, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 1, size: 8, align: 8, elements: !13, identifier: "_ZTS1A")
!13 = !{!14, !21, !22}
!14 = !DISubprogram(name: "A", scope: !12, file: !1, line: 2, type: !15, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !18}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64, align: 64)
!19 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !20)
!20 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!21 = !DISubprogram(name: "operator+=", linkageName: "_ZN1ApLEPKc", scope: !12, file: !1, line: 5, type: !15, isLocal: false, isDefinition: false, scopeLine: 5, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!22 = !DISubprogram(name: "m_fn1", linkageName: "_ZN1A5m_fn1ES_", scope: !12, file: !1, line: 6, type: !23, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !17, !12}
!25 = !DILocalVariable(name: "opts", scope: !6, file: !1, line: 20, type: !26)
!26 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!27 = !DILocalVariable(name: "indent", scope: !6, file: !1, line: 20, type: !26)
!28 = !DILocalVariable(name: "type", scope: !6, file: !1, line: 20, type: !26)
!29 = !DILocalVariable(name: "v", scope: !6, file: !1, line: 21, type: !30)
!30 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !1, line: 9, size: 8, align: 8, elements: !31, identifier: "_ZTS1B")
!31 = !{!32}
!32 = !DISubprogram(name: "operator[]", linkageName: "_ZN1BixEj", scope: !30, file: !1, line: 11, type: !33, isLocal: false, isDefinition: false, scopeLine: 11, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !36, !37}
!35 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !30, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!37 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!38 = !DILocalVariable(name: "i", scope: !39, file: !1, line: 22, type: !26)
!39 = distinct !DILexicalBlock(scope: !6, file: !1, line: 22, column: 3)
!40 = !DILocation(line: 21, column: 3, scope: !6)
!41 = !DILocation(line: 14, column: 3, scope: !42, inlinedAt: !56)
!42 = distinct !DISubprogram(name: "Print<double>", linkageName: "_Z5PrintIdEvT_iiPiiP1A", scope: !1, file: !1, line: 13, type: !43, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !46, variables: !48)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !35, !26, !26, !45, !26, !11}
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64, align: 64)
!46 = !{!47}
!47 = !DITemplateTypeParameter(name: "T", type: !35)
!48 = !{!49, !50, !51, !52, !53, !54, !55}
!49 = !DILocalVariable(name: "p1", arg: 1, scope: !42, file: !1, line: 13, type: !35)
!50 = !DILocalVariable(arg: 2, scope: !42, file: !1, line: 13, type: !26)
!51 = !DILocalVariable(arg: 3, scope: !42, file: !1, line: 13, type: !26)
!52 = !DILocalVariable(arg: 4, scope: !42, file: !1, line: 13, type: !45)
!53 = !DILocalVariable(arg: 5, scope: !42, file: !1, line: 13, type: !26)
!54 = !DILocalVariable(name: "p6", arg: 6, scope: !42, file: !1, line: 13, type: !11)
!55 = !DILocalVariable(name: "text", scope: !42, file: !1, line: 14, type: !12)
!56 = distinct !DILocation(line: 23, column: 5, scope: !57)
!57 = !DILexicalBlockFile(scope: !58, file: !1, discriminator: 1)
!58 = distinct !DILexicalBlock(scope: !39, file: !1, line: 22, column: 3)
!59 = !DILocation(line: 0, scope: !60, inlinedAt: !64)
!60 = distinct !DISubprogram(name: "operator+=", linkageName: "_ZN1ApLEPKc", scope: !12, file: !1, line: 5, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !21, variables: !61)
!61 = !{!62, !63}
!62 = !DILocalVariable(name: "this", arg: 1, scope: !60, type: !11, flags: DIFlagArtificial | DIFlagObjectPointer)
!63 = !DILocalVariable(name: "p1", arg: 2, scope: !60, file: !1, line: 5, type: !18)
!64 = distinct !DILocation(line: 15, column: 8, scope: !42, inlinedAt: !56)
!65 = !DILocation(line: 22, column: 8, scope: !39)
!66 = !DIExpression(DW_OP_deref)
!67 = !DILocation(line: 21, column: 5, scope: !6)
!68 = !DILocation(line: 23, column: 11, scope: !58)
!69 = !DIExpression()
!70 = !DILocation(line: 13, column: 36, scope: !42, inlinedAt: !56)
!71 = !DILocation(line: 13, column: 55, scope: !42, inlinedAt: !56)
!72 = !DILocation(line: 13, column: 65, scope: !42, inlinedAt: !56)
!73 = !DILocation(line: 15, column: 11, scope: !42, inlinedAt: !56)
!74 = !DILocation(line: 14, column: 5, scope: !42, inlinedAt: !56)
!75 = !DILocation(line: 5, column: 31, scope: !60, inlinedAt: !64)
!76 = !DILocation(line: 5, column: 43, scope: !60, inlinedAt: !64)
!77 = !DILocation(line: 5, column: 37, scope: !78, inlinedAt: !64)
!78 = !DILexicalBlockFile(scope: !60, file: !1, discriminator: 1)
!79 = !DILocation(line: 5, column: 48, scope: !60, inlinedAt: !64)
!80 = !DILocation(line: 16, column: 1, scope: !42, inlinedAt: !56)
!81 = !DILocation(line: 22, column: 3, scope: !57)
!82 = distinct !{!82, !83}
!83 = !DILocation(line: 22, column: 3, scope: !6)
