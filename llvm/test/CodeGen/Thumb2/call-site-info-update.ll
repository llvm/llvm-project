; RUN: llc < %s -filetype=obj -debug-entry-values

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-unknown-linux-gnueabi"

%class.z = type { i8, i8 }
%struct.y = type { %class.ac }
%class.ac = type { %"struct.ac::m" }
%"struct.ac::m" = type { i8 }
%class.ae = type { i8 }
%class.o = type { i8 }

@l = dso_local local_unnamed_addr global i8 0, align 1, !dbg !0

; Function Attrs: nounwind
define dso_local zeroext i1 @_ZN1z2agEv(ptr nocapture readonly %this) local_unnamed_addr align 2 !dbg !17 {
entry:
  %a = alloca %struct.y, align 1
  %b = alloca %struct.y, align 1
  %c = alloca %class.ae, align 1
  call void @llvm.dbg.value(metadata ptr %this, metadata !28, metadata !DIExpression()), !dbg !75
  %0 = load i8, ptr %this, align 1, !dbg !75
  %tobool = icmp eq i8 %0, 0, !dbg !75
  br i1 %tobool, label %if.end, label %if.then, !dbg !75

if.then:                                          ; preds = %entry
  %call.i.i = call ptr @_ZN2ac1mC1EPc(ptr nonnull %a, ptr null), !dbg !82
  %call2.i.i = call ptr @_ZN2ac2adEv(ptr nonnull %a), !dbg !82
  %cmp.i.i = icmp eq ptr %call2.i.i, null, !dbg !82
  %frombool.i.i = zext i1 %cmp.i.i to i8, !dbg !82
  store i8 %frombool.i.i, ptr @l, align 1, !dbg !82
  br i1 %cmp.i.i, label %_ZN1yC2Ev.exit, label %if.then.i.i, !dbg !82

if.then.i.i:                                      ; preds = %if.then
  call void @llvm.dbg.value(metadata i32 1, metadata !144, metadata !DIExpression()), !dbg !145
  call void @_ZdlPv(ptr null), !dbg !145
  br label %_ZN1yC2Ev.exit, !dbg !145

_ZN1yC2Ev.exit:                                   ; preds = %if.then, %if.then.i.i
  call void @llvm.dbg.value(metadata i8 1, metadata !31, metadata !DIExpression()), !dbg !75
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %a), !dbg !75
  br label %cleanup

if.end:                                           ; preds = %entry
  %ah = getelementptr inbounds %class.z, ptr %this, i32 0, i32 1, !dbg !150
  %1 = load i8, ptr %ah, align 1, !dbg !150
  %tobool3 = icmp eq i8 %1, 0, !dbg !150
  br i1 %tobool3, label %if.end7, label %if.then4, !dbg !150

if.then4:                                         ; preds = %if.end
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %b), !dbg !153
  %call.i.i13 = call ptr @_ZN2ac1mC1EPc(ptr nonnull %b, ptr null), !dbg !153
  %call2.i.i14 = call ptr @_ZN2ac2adEv(ptr nonnull %b), !dbg !153
  %cmp.i.i15 = icmp eq ptr %call2.i.i14, null, !dbg !153
  %frombool.i.i16 = zext i1 %cmp.i.i15 to i8, !dbg !153
  store i8 %frombool.i.i16, ptr @l, align 1, !dbg !153
  br i1 %cmp.i.i15, label %_ZN1yC2Ev.exit18, label %if.then.i.i17, !dbg !153

if.then.i.i17:                                    ; preds = %if.then4
  call void @_ZdlPv(ptr null), !dbg !170
  br label %_ZN1yC2Ev.exit18, !dbg !170

_ZN1yC2Ev.exit18:                                 ; preds = %if.then4, %if.then.i.i17
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %b), !dbg !173
  br label %cleanup

if.end7:                                          ; preds = %if.end
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %c), !dbg !173
  %call8 = call ptr @_ZN2aeC1Ei(ptr nonnull %c, i32 1), !dbg !173
  call void @_ZN2ae1xES_(ptr nonnull %c, [1 x i32] zeroinitializer), !dbg !173
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %c), !dbg !173
  br label %cleanup

cleanup:                                          ; preds = %if.end7, %_ZN1yC2Ev.exit18, %_ZN1yC2Ev.exit
  %retval.0 = phi i1 [ true, %_ZN1yC2Ev.exit ], [ true, %_ZN1yC2Ev.exit18 ], [ false, %if.end7 ], !dbg !75
  ret i1 %retval.0, !dbg !75
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare dso_local ptr @_ZN2aeC1Ei(ptr returned, i32) unnamed_addr

declare dso_local void @_ZN2ae1xES_(ptr, [1 x i32]) local_unnamed_addr

declare dso_local ptr @_ZN2ac1mC1EPc(ptr returned, ptr) unnamed_addr

declare dso_local ptr @_ZN2ac2adEv(ptr) local_unnamed_addr

; Function Attrs: nobuiltin nounwind
declare !dbg !6 dso_local void @_ZdlPv(ptr) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "l", scope: !2, file: !3, line: 14, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !10, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "crash.cpp", directory: "/")
!4 = !{}
!5 = !{!6}
!6 = !DISubprogram(name: "operator delete", linkageName: "_ZdlPv", scope: !3, file: !3, type: !7, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!10 = !{!0}
!11 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 1, !"min_enum_size", i32 4}
!16 = !{!"clang version 11.0.0"}
!17 = distinct !DISubprogram(name: "ag", linkageName: "_ZN1z2agEv", scope: !18, file: !3, line: 45, type: !24, scopeLine: 45, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !23, retainedNodes: !27)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "z", file: !3, line: 40, size: 16, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTS1z")
!19 = !{!20, !22, !23}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !18, file: !3, line: 42, baseType: !21, size: 8)
!21 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "ah", scope: !18, file: !3, line: 43, baseType: !21, size: 8, offset: 8)
!23 = !DISubprogram(name: "ag", linkageName: "_ZN1z2agEv", scope: !18, file: !3, line: 41, type: !24, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(types: !25)
!25 = !{!11, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !{!28, !30, !31, !32, !55, !58}
!28 = !DILocalVariable(name: "this", arg: 1, scope: !17, type: !29, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 32)
!30 = !DILocalVariable(name: "w", scope: !17, file: !3, line: 46, type: !11)
!31 = !DILocalVariable(name: "v", scope: !17, file: !3, line: 46, type: !11)
!32 = !DILocalVariable(name: "a", scope: !33, file: !3, line: 48, type: !35)
!33 = distinct !DILexicalBlock(scope: !34, file: !3, line: 47, column: 10)
!34 = distinct !DILexicalBlock(scope: !17, file: !3, line: 47, column: 7)
!35 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "y", file: !3, line: 37, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !36, identifier: "_ZTS1y")
!36 = !{!37}
!37 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !35, file: !3, line: 38, baseType: !38, size: 8)
!38 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ac", file: !3, line: 15, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !39, identifier: "_ZTS2ac")
!39 = !{!40, !48, !52}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !38, file: !3, line: 18, baseType: !41, size: 8)
!41 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "m", scope: !38, file: !3, line: 16, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !42, identifier: "_ZTSN2ac1mE")
!42 = !{!43}
!43 = !DISubprogram(name: "m", scope: !41, file: !3, line: 17, type: !44, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !46, !47}
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 32)
!48 = !DISubprogram(name: "ad", linkageName: "_ZN2ac2adEv", scope: !38, file: !3, line: 19, type: !49, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!49 = !DISubroutineType(types: !50)
!50 = !{!47, !51}
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!52 = !DISubprogram(name: "ac", scope: !38, file: !3, line: 22, type: !53, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!53 = !DISubroutineType(types: !54)
!54 = !{null, !51}
!55 = !DILocalVariable(name: "b", scope: !56, file: !3, line: 53, type: !35)
!56 = distinct !DILexicalBlock(scope: !57, file: !3, line: 52, column: 11)
!57 = distinct !DILexicalBlock(scope: !17, file: !3, line: 52, column: 7)
!58 = !DILocalVariable(name: "c", scope: !17, file: !3, line: 57, type: !59)
!59 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ae", file: !3, line: 30, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !60, identifier: "_ZTS2ae")
!60 = !{!61, !62, !68, !72}
!61 = !DIDerivedType(tag: DW_TAG_member, name: "af", scope: !59, file: !3, line: 32, baseType: !21, size: 8, flags: DIFlagPublic)
!62 = !DISubprogram(name: "ae", scope: !59, file: !3, line: 33, type: !63, scopeLine: 33, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!63 = !DISubroutineType(types: !64)
!64 = !{null, !65, !66}
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !59, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 32)
!67 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!68 = !DISubprogram(name: "ae", scope: !59, file: !3, line: 34, type: !69, scopeLine: 34, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!69 = !DISubroutineType(types: !70)
!70 = !{null, !65, !71}
!71 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!72 = !DISubprogram(name: "x", linkageName: "_ZN2ae1xES_", scope: !59, file: !3, line: 35, type: !73, scopeLine: 35, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!73 = !DISubroutineType(types: !74)
!74 = !{null, !65, !59}
!75 = !DILocation(line: 0, scope: !17)
!82 = !DILocation(line: 48, column: 5, scope: !33)
!84 = !DILocalVariable(name: "this", arg: 1, scope: !85, type: !91, flags: DIFlagArtificial | DIFlagObjectPointer)
!85 = distinct !DISubprogram(name: "y", linkageName: "_ZN1yC2Ev", scope: !35, file: !3, line: 37, type: !86, scopeLine: 37, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !89, retainedNodes: !90)
!86 = !DISubroutineType(types: !87)
!87 = !{null, !88}
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!89 = !DISubprogram(name: "y", scope: !35, type: !86, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!90 = !{!84}
!91 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 32)
!93 = distinct !DILocation(line: 48, column: 7, scope: !33)
!94 = !DILocation(line: 37, column: 8, scope: !85, inlinedAt: !93)
!95 = !DILocalVariable(name: "this", arg: 1, scope: !96, type: !114, flags: DIFlagArtificial | DIFlagObjectPointer)
!96 = distinct !DISubprogram(name: "ac", linkageName: "_ZN2acC2Ev", scope: !38, file: !3, line: 22, type: !53, scopeLine: 22, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !52, retainedNodes: !97)
!97 = !{!95, !98}
!98 = !DILocalVariable(name: "t", scope: !99, file: !3, line: 25, type: !102)
!99 = distinct !DILexicalBlock(scope: !100, file: !3, line: 24, column: 13)
!100 = distinct !DILexicalBlock(scope: !101, file: !3, line: 24, column: 9)
!101 = distinct !DILexicalBlock(scope: !96, file: !3, line: 22, column: 15)
!102 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "o<char>", file: !3, line: 6, size: 8, flags: DIFlagTypePassByValue, elements: !103, templateParams: !112, identifier: "_ZTS1oIcE")
!103 = !{!104}
!104 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !102, baseType: !105, flags: DIFlagPublic, extraData: i32 0)
!105 = !DIDerivedType(tag: DW_TAG_typedef, name: "d<char>", file: !3, line: 5, baseType: !106)
!106 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "i", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !107, identifier: "_ZTS1i")
!107 = !{!108}
!108 = !DISubprogram(name: "j", linkageName: "_ZN1i1jEPci", scope: !106, file: !3, line: 3, type: !109, scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!109 = !DISubroutineType(types: !110)
!110 = !{null, !111, !47, !71}
!111 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !106, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!112 = !{!113}
!113 = !DITemplateTypeParameter(type: !21)
!114 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 32)
!116 = distinct !DILocation(line: 37, column: 8, scope: !85, inlinedAt: !93)
!124 = !DILocalVariable(name: "k", arg: 1, scope: !125, file: !3, line: 10, type: !102)
!125 = distinct !DISubprogram(name: "p", linkageName: "_ZN2aaI1oIcEE1pES1_Pci", scope: !126, file: !3, line: 10, type: !129, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !128, retainedNodes: !133)
!126 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "aa<o<char> >", file: !3, line: 9, size: 8, flags: DIFlagTypePassByValue, elements: !127, templateParams: !131, identifier: "_ZTS2aaI1oIcEE")
!127 = !{!128}
!128 = !DISubprogram(name: "p", linkageName: "_ZN2aaI1oIcEE1pES1_Pci", scope: !126, file: !3, line: 10, type: !129, scopeLine: 10, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagOptimized)
!129 = !DISubroutineType(types: !130)
!130 = !{null, !102, !47, !71}
!131 = !{!132}
!132 = !DITemplateTypeParameter(type: !102)
!133 = !{!124, !134, !135}
!134 = !DILocalVariable(name: "ab", arg: 2, scope: !125, file: !3, line: 10, type: !47)
!135 = !DILocalVariable(name: "q", arg: 3, scope: !125, file: !3, line: 10, type: !71)
!136 = !DILocation(line: 10, column: 22, scope: !125, inlinedAt: !137)
!137 = distinct !DILocation(line: 26, column: 7, scope: !99, inlinedAt: !116)
!138 = !DILocation(line: 0, scope: !125, inlinedAt: !137)
!139 = !DILocalVariable(name: "k", arg: 2, scope: !140, file: !3, line: 3, type: !47)
!140 = distinct !DISubprogram(name: "j", linkageName: "_ZN1i1jEPci", scope: !106, file: !3, line: 3, type: !109, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !108, retainedNodes: !141)
!141 = !{!142, !139, !144}
!142 = !DILocalVariable(name: "this", arg: 1, scope: !140, type: !143, flags: DIFlagArtificial | DIFlagObjectPointer)
!143 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !106, size: 32)
!144 = !DILocalVariable(arg: 3, scope: !140, file: !3, line: 3, type: !71)
!145 = !DILocation(line: 0, scope: !140, inlinedAt: !146)
!146 = distinct !DILocation(line: 10, column: 43, scope: !125, inlinedAt: !137)
!150 = !DILocation(line: 52, column: 7, scope: !57)
!153 = !DILocation(line: 53, column: 5, scope: !56)
!156 = distinct !DILocation(line: 53, column: 7, scope: !56)
!159 = distinct !DILocation(line: 37, column: 8, scope: !85, inlinedAt: !156)
!166 = distinct !DILocation(line: 26, column: 7, scope: !99, inlinedAt: !159)
!167 = !DILocation(line: 0, scope: !125, inlinedAt: !166)
!169 = distinct !DILocation(line: 10, column: 43, scope: !125, inlinedAt: !166)
!170 = !DILocation(line: 3, column: 26, scope: !140, inlinedAt: !169)
!173 = !DILocation(line: 57, column: 3, scope: !17)
