; RUN: llc --stop-after=finalize-isel < %s

; Tests that files with multiple SDNode dependencies are correctly handled by
; SelectionDAG; dependencies that are incorrectly updated for an SDDbgValue
; throughout SelectionDAG will result in errors.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.anon = type { i8 }
%class.D = type { %class.f }
%class.f = type { i32, i32 }
%class.l = type { i8*, i64 }
%class.d = type { i8 }

@q = internal global %class.anon zeroinitializer, align 1, !dbg !0

define void @_Z1rv() local_unnamed_addr !dbg !30 {
entry:
  %t = alloca %class.D, align 4
  %call = tail call { i8*, i64 } @"_ZN3$_05m_fn3Ev"(%class.anon* nonnull dereferenceable(1) @q), !dbg !60
  %0 = extractvalue { i8*, i64 } %call, 0, !dbg !60
  call void @llvm.dbg.value(metadata i8* %0, metadata !34, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !61
  %1 = extractvalue { i8*, i64 } %call, 1, !dbg !60
  call void @llvm.dbg.value(metadata i64 %1, metadata !34, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !61
  %2 = bitcast %class.D* %t to i8*, !dbg !62
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2) #4, !dbg !62
  call void @llvm.dbg.declare(metadata %class.D* %t, metadata !35, metadata !DIExpression()), !dbg !63
  call void @llvm.dbg.value(metadata i8* %0, metadata !64, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !69
  call void @llvm.dbg.value(metadata i64 %1, metadata !64, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !69
  call void @llvm.dbg.value(metadata %class.D* %t, metadata !67, metadata !DIExpression()), !dbg !69
  call void @llvm.dbg.value(metadata %class.l* undef, metadata !71, metadata !DIExpression()), !dbg !75
  call void @llvm.dbg.value(metadata %class.D* %t, metadata !77, metadata !DIExpression()), !dbg !91
  call void @llvm.dbg.value(metadata i8* %0, metadata !86, metadata !DIExpression()), !dbg !91
  call void @llvm.dbg.value(metadata !DIArgList(i8* %0, i64 %1), metadata !87, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_stack_value)), !dbg !91
  %3 = bitcast %class.D* %t to %class.d*, !dbg !93
  call void @_ZN1dC2EPvl(%class.d* nonnull dereferenceable(1) %3, i8* nonnull %2, i64 8), !dbg !94
  call void @llvm.dbg.value(metadata !DIArgList(i8* %0, i8* %0, i64 %1), metadata !88, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 2, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !95
  %tobool.not.i.i = icmp eq i64 %1, 0, !dbg !96
  br i1 %tobool.not.i.i, label %_ZN1DILi8EEC2E1l.exit, label %if.then.i.i, !dbg !98

if.then.i.i:                                      ; preds = %entry
  %add.ptr.i.i = getelementptr inbounds i8, i8* %0, i64 %1, !dbg !99
  call void @llvm.dbg.value(metadata i8* %add.ptr.i.i, metadata !87, metadata !DIExpression()), !dbg !91
  call void @llvm.dbg.value(metadata !DIArgList(i8* %add.ptr.i.i, i8* %0), metadata !88, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !95
  call void @llvm.dbg.value(metadata %class.D* %t, metadata !77, metadata !DIExpression()), !dbg !91
  %k.i.i = getelementptr inbounds %class.D, %class.D* %t, i64 0, i32 0, i32 1, !dbg !100
  %4 = load i32, i32* %k.i.i, align 4, !dbg !100, !tbaa !101
  call void @_Z1cIPKciEvT_S2_T0_(i8* %0, i8* nonnull %add.ptr.i.i, i32 %4), !dbg !106
  br label %_ZN1DILi8EEC2E1l.exit, !dbg !106

_ZN1DILi8EEC2E1l.exit:                            ; preds = %entry, %if.then.i.i
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2) #4, !dbg !107
  ret void, !dbg !107
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare { i8*, i64 } @"_ZN3$_05m_fn3Ev"(%class.anon* nonnull dereferenceable(1)) local_unnamed_addr

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

declare void @_ZN1dC2EPvl(%class.d* nonnull dereferenceable(1), i8*, i64) unnamed_addr

declare !dbg !108 void @_Z1cIPKciEvT_S2_T0_(i8*, i8*, i32) local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!26, !27, !28}
!llvm.ident = !{!29}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 32, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "dbg-list-dependencies.cpp", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, file: !3, line: 29, size: 8, flags: DIFlagTypePassByValue, elements: !7)
!7 = !{!8}
!8 = !DISubprogram(name: "m_fn3", linkageName: "_ZN3$_05m_fn3Ev", scope: !6, file: !3, line: 31, type: !9, scopeLine: 31, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !25}
!11 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "l", file: !3, line: 17, size: 128, flags: DIFlagTypePassByValue, elements: !12, identifier: "_ZTS1l")
!12 = !{!13, !16, !18, !24}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !11, file: !3, line: 18, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !11, file: !3, line: 19, baseType: !17, size: 64, offset: 64)
!17 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!18 = !DISubprogram(name: "p", linkageName: "_ZN1l1pEv", scope: !11, file: !3, line: 22, type: !19, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!19 = !DISubroutineType(types: !20)
!20 = !{!21, !23}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !15)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DISubprogram(name: "m_fn2", linkageName: "_ZN1l5m_fn2Ev", scope: !11, file: !3, line: 23, type: !19, scopeLine: 23, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!26 = !{i32 7, !"Dwarf Version", i32 4}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 1, !"wchar_size", i32 4}
!29 = !{!"clang version 13.0.0"}
!30 = distinct !DISubprogram(name: "r", linkageName: "_Z1rv", scope: !3, file: !3, line: 33, type: !31, scopeLine: 33, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{null}
!33 = !{!34, !35}
!34 = !DILocalVariable(name: "s", scope: !30, file: !3, line: 34, type: !11)
!35 = !DILocalVariable(name: "t", scope: !30, file: !3, line: 35, type: !36)
!36 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "D<8>", file: !3, line: 25, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !37, templateParams: !58, identifier: "_ZTS1DILi8EE")
!37 = !{!38, !54}
!38 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !36, baseType: !39, extraData: i32 0)
!39 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "f<8>", file: !3, line: 6, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !40, templateParams: !52, identifier: "_ZTS1fILi8EE")
!40 = !{!41, !49, !51}
!41 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !39, baseType: !42, extraData: i32 0)
!42 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "d", file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !43, identifier: "_ZTS1d")
!43 = !{!44}
!44 = !DISubprogram(name: "d", scope: !42, file: !3, line: 4, type: !45, scopeLine: 4, flags: DIFlagProtected | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!45 = !DISubroutineType(types: !46)
!46 = !{null, !47, !48, !17}
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !42, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !39, file: !3, line: 7, baseType: !50, size: 32)
!50 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!51 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !39, file: !3, line: 15, baseType: !50, size: 32, offset: 32, flags: DIFlagPublic)
!52 = !{!53}
!53 = !DITemplateValueParameter(name: "e", type: !50, value: i32 8)
!54 = !DISubprogram(name: "D", scope: !36, file: !3, line: 27, type: !55, scopeLine: 27, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!55 = !DISubroutineType(types: !56)
!56 = !{null, !57, !11}
!57 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!58 = !{!59}
!59 = !DITemplateValueParameter(name: "o", type: !50, value: i32 8)
!60 = !DILocation(line: 34, column: 9, scope: !30)
!61 = !DILocation(line: 0, scope: !30)
!62 = !DILocation(line: 35, column: 3, scope: !30)
!63 = !DILocation(line: 35, column: 8, scope: !30)
!64 = !DILocalVariable(name: "h", arg: 2, scope: !65, file: !3, line: 27, type: !11)
!65 = distinct !DISubprogram(name: "D", linkageName: "_ZN1DILi8EEC2E1l", scope: !36, file: !3, line: 27, type: !55, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !54, retainedNodes: !66)
!66 = !{!67, !64}
!67 = !DILocalVariable(name: "this", arg: 1, scope: !65, type: !68, flags: DIFlagArtificial | DIFlagObjectPointer)
!68 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64)
!69 = !DILocation(line: 0, scope: !65, inlinedAt: !70)
!70 = distinct !DILocation(line: 35, column: 8, scope: !30)
!71 = !DILocalVariable(name: "this", arg: 1, scope: !72, type: !74, flags: DIFlagArtificial | DIFlagObjectPointer)
!72 = distinct !DISubprogram(name: "m_fn2", linkageName: "_ZN1l5m_fn2Ev", scope: !11, file: !3, line: 23, type: !19, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !24, retainedNodes: !73)
!73 = !{!71}
!74 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!75 = !DILocation(line: 0, scope: !72, inlinedAt: !76)
!76 = distinct !DILocation(line: 27, column: 26, scope: !65, inlinedAt: !70)
!77 = !DILocalVariable(name: "this", arg: 1, scope: !78, type: !90, flags: DIFlagArtificial | DIFlagObjectPointer)
!78 = distinct !DISubprogram(name: "f<const char *>", linkageName: "_ZN1fILi8EEC2IPKcEET_S4_", scope: !39, file: !3, line: 10, type: !79, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, templateParams: !83, declaration: !82, retainedNodes: !85)
!79 = !DISubroutineType(types: !80)
!80 = !{null, !81, !21, !21}
!81 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!82 = !DISubprogram(name: "f<const char *>", scope: !39, file: !3, line: 10, type: !79, scopeLine: 10, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized, templateParams: !83)
!83 = !{!84}
!84 = !DITemplateTypeParameter(name: "a", type: !21)
!85 = !{!77, !86, !87, !88}
!86 = !DILocalVariable(name: "h", arg: 2, scope: !78, file: !3, line: 10, type: !21)
!87 = !DILocalVariable(name: "i", arg: 3, scope: !78, file: !3, line: 10, type: !21)
!88 = !DILocalVariable(name: "j", scope: !89, file: !3, line: 11, type: !17)
!89 = distinct !DILexicalBlock(scope: !78, file: !3, line: 10, column: 48)
!90 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64)
!91 = !DILocation(line: 0, scope: !78, inlinedAt: !92)
!92 = distinct !DILocation(line: 27, column: 12, scope: !65, inlinedAt: !70)
!93 = !DILocation(line: 10, column: 48, scope: !78, inlinedAt: !92)
!94 = !DILocation(line: 10, column: 39, scope: !78, inlinedAt: !92)
!95 = !DILocation(line: 0, scope: !89, inlinedAt: !92)
!96 = !DILocation(line: 12, column: 9, scope: !97, inlinedAt: !92)
!97 = distinct !DILexicalBlock(scope: !89, file: !3, line: 12, column: 9)
!98 = !DILocation(line: 12, column: 9, scope: !89, inlinedAt: !92)
!99 = !DILocation(line: 23, column: 34, scope: !72, inlinedAt: !76)
!100 = !DILocation(line: 13, column: 15, scope: !97, inlinedAt: !92)
!101 = !{!102, !103, i64 4}
!102 = !{!"_ZTS1fILi8EE", !103, i64 0, !103, i64 4}
!103 = !{!"int", !104, i64 0}
!104 = !{!"omnipotent char", !105, i64 0}
!105 = !{!"Simple C++ TBAA"}
!106 = !DILocation(line: 13, column: 7, scope: !97, inlinedAt: !92)
!107 = !DILocation(line: 36, column: 1, scope: !30)
!108 = !DISubprogram(name: "c<const char *, int>", linkageName: "_Z1cIPKciEvT_S2_T0_", scope: !3, file: !3, line: 1, type: !109, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, templateParams: !111, retainedNodes: !4)
!109 = !DISubroutineType(types: !110)
!110 = !{null, !21, !21, !50}
!111 = !{!84, !112}
!112 = !DITemplateTypeParameter(name: "b", type: !50)
