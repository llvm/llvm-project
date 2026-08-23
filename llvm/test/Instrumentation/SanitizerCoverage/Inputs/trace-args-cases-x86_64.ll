; A set of C and C++ argument shapes, as clang -O2 -g emits them
; for x86_64-linux-gnu: cases.c and cases.cpp, linked into one module.
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.big = type { i64, i64, i64, i64 }

define dso_local i64 @c01_scalar(i32 noundef %a, i64 noundef %b) local_unnamed_addr !dbg !23 {
entry:
    #dbg_value(i32 %a, !28, !DIExpression(), !30)
    #dbg_value(i64 %b, !29, !DIExpression(), !30)
  %conv = sext i32 %a to i64, !dbg !31
  %add = add nsw i64 %b, %conv, !dbg !32
  ret i64 %add, !dbg !33
}

define dso_local noundef i64 @c02_pointer(ptr noundef %p) local_unnamed_addr !dbg !34 {
entry:
    #dbg_value(ptr %p, !40, !DIExpression(), !41)
  %0 = ptrtoint ptr %p to i64, !dbg !42
  ret i64 %0, !dbg !43
}

define dso_local noundef double @c03_fp(float noundef %f, double noundef %d) local_unnamed_addr !dbg !44 {
entry:
    #dbg_value(float %f, !50, !DIExpression(), !52)
    #dbg_value(double %d, !51, !DIExpression(), !52)
  %conv = fpext float %f to double, !dbg !53
  %add = fadd double %d, %conv, !dbg !54
  ret double %add, !dbg !55
}

define dso_local i32 @c04_small(i64 %p.coerce) local_unnamed_addr !dbg !56 {
entry:
  %p.sroa.0.0.extract.trunc = trunc i64 %p.coerce to i32
  %p.sroa.2.0.extract.shift = lshr i64 %p.coerce, 32
  %p.sroa.2.0.extract.trunc = trunc nuw i64 %p.sroa.2.0.extract.shift to i32
    #dbg_value(i32 %p.sroa.0.0.extract.trunc, !64, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !65)
    #dbg_value(i32 %p.sroa.2.0.extract.trunc, !64, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !65)
  %add = add nsw i32 %p.sroa.2.0.extract.trunc, %p.sroa.0.0.extract.trunc, !dbg !66
  ret i32 %add, !dbg !67
}

define dso_local double @c05_split(double %m.coerce0, i64 %m.coerce1) local_unnamed_addr !dbg !68 {
entry:
    #dbg_value(double %m.coerce0, !76, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !77)
    #dbg_value(i64 %m.coerce1, !76, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !77)
  %conv = sitofp i64 %m.coerce1 to double, !dbg !78
  %add = fadd double %m.coerce0, %conv, !dbg !79
  ret double %add, !dbg !80
}

define dso_local float @c05_hfa(<2 x float> %h.coerce0, float %h.coerce1) local_unnamed_addr !dbg !81 {
entry:
    #dbg_value(<2 x float> %h.coerce0, !90, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !91)
    #dbg_value(float %h.coerce1, !90, !DIExpression(DW_OP_LLVM_fragment, 64, 32), !91)
  %shift = shufflevector <2 x float> %h.coerce0, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !92
  %foldExtExtBinop = fadd <2 x float> %h.coerce0, %shift, !dbg !92
  %add = extractelement <2 x float> %foldExtExtBinop, i64 0, !dbg !92
  %add1 = fadd float %h.coerce1, %add, !dbg !93
  ret float %add1, !dbg !94
}

define dso_local i64 @c06_big(ptr nofree noundef readonly byval(%struct.big) align 8 captures(none) %b) local_unnamed_addr !dbg !95 {
entry:
    #dbg_declare(ptr %b, !105, !DIExpression(), !106)
  %0 = load i64, ptr %b, align 8, !dbg !107, !tbaa !108
  %d = getelementptr inbounds nuw i8, ptr %b, i64 24, !dbg !111
  %1 = load i64, ptr %d, align 8, !dbg !111, !tbaa !112
  %add = add nsw i64 %1, %0, !dbg !113
  ret i64 %add, !dbg !114
}

define dso_local void @c07_sret(ptr dead_on_unwind noalias nofree writable writeonly sret(%struct.big) align 8 captures(none) initializes((0, 32)) %agg.result, i64 noundef %x) local_unnamed_addr !dbg !115 {
entry:
    #dbg_value(i64 %x, !119, !DIExpression(), !121)
    #dbg_declare(ptr %agg.result, !120, !DIExpression(), !122)
  %0 = insertelement <4 x i64> poison, i64 %x, i64 0, !dbg !123
  %1 = shufflevector <4 x i64> %0, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !123
  %2 = add nsw <4 x i64> %1, <i64 0, i64 1, i64 2, i64 3>, !dbg !123
  store <4 x i64> %2, ptr %agg.result, align 8, !dbg !124, !tbaa !125
  ret void, !dbg !126
}

define dso_local i64 @c08_struct_ptr(ptr nofree noundef readonly captures(none) %f) local_unnamed_addr !dbg !127 {
entry:
    #dbg_value(ptr %f, !138, !DIExpression(), !139)
  %0 = load i32, ptr %f, align 8, !dbg !140, !tbaa !141
  %conv = sext i32 %0 to i64, !dbg !143
  %b = getelementptr inbounds nuw i8, ptr %f, i64 8, !dbg !144
  %1 = load i64, ptr %b, align 8, !dbg !144, !tbaa !145
  %add = add nsw i64 %1, %conv, !dbg !146
  %c = getelementptr inbounds nuw i8, ptr %f, i64 16, !dbg !147
  %2 = load i8, ptr %c, align 8, !dbg !147, !tbaa !148
  %conv1 = sext i8 %2 to i64, !dbg !149
  %add2 = add nsw i64 %add, %conv1, !dbg !150
  ret i64 %add2, !dbg !151
}

define dso_local i64 @c09_stack(i64 noundef %a0, i64 noundef %a1, i64 noundef %a2, i64 noundef %a3, i64 noundef %a4, i64 noundef %a5, i64 noundef %a6, i64 noundef %a7, i64 noundef %a8, i64 noundef %a9) local_unnamed_addr !dbg !152 {
entry:
    #dbg_value(i64 %a0, !156, !DIExpression(), !166)
    #dbg_value(i64 %a1, !157, !DIExpression(), !166)
    #dbg_value(i64 %a2, !158, !DIExpression(), !166)
    #dbg_value(i64 %a3, !159, !DIExpression(), !166)
    #dbg_value(i64 %a4, !160, !DIExpression(), !166)
    #dbg_value(i64 %a5, !161, !DIExpression(), !166)
    #dbg_value(i64 %a6, !162, !DIExpression(), !166)
    #dbg_value(i64 %a7, !163, !DIExpression(), !166)
    #dbg_value(i64 %a8, !164, !DIExpression(), !166)
    #dbg_value(i64 %a9, !165, !DIExpression(), !166)
  %add = add nsw i64 %a1, %a0, !dbg !167
  %add1 = add nsw i64 %add, %a2, !dbg !168
  %add2 = add nsw i64 %add1, %a3, !dbg !169
  %add3 = add nsw i64 %add2, %a4, !dbg !170
  %add4 = add nsw i64 %add3, %a5, !dbg !171
  %add5 = add nsw i64 %add4, %a6, !dbg !172
  %add6 = add nsw i64 %add5, %a7, !dbg !173
  %add7 = add nsw i64 %add6, %a8, !dbg !174
  %add8 = add nsw i64 %add7, %a9, !dbg !175
  ret i64 %add8, !dbg !176
}

define dso_local noundef double @c10_mixed(i32 noundef %i, double noundef %d, i64 noundef %l, float noundef %f) local_unnamed_addr !dbg !177 {
entry:
    #dbg_value(i32 %i, !181, !DIExpression(), !185)
    #dbg_value(double %d, !182, !DIExpression(), !185)
    #dbg_value(i64 %l, !183, !DIExpression(), !185)
    #dbg_value(float %f, !184, !DIExpression(), !185)
  %conv = sitofp i32 %i to double, !dbg !186
  %add = fadd double %d, %conv, !dbg !187
  %conv1 = sitofp i64 %l to double, !dbg !188
  %add2 = fadd double %add, %conv1, !dbg !189
  %conv3 = fpext float %f to double, !dbg !190
  %add4 = fadd double %add2, %conv3, !dbg !191
  ret double %add4, !dbg !192
}

define dso_local range(i64 -9223372036854775807, -9223372036854775808) i64 @c12_outer(ptr nofree noundef readonly captures(none) %p) local_unnamed_addr !dbg !193 {
entry:
    #dbg_value(ptr %p, !195, !DIExpression(), !196)
    #dbg_value(ptr %p, !197, !DIExpression(), !200)
  %0 = load i32, ptr %p, align 8, !dbg !202, !tbaa !141
  %mul.i = mul nsw i32 %0, 3, !dbg !203
  %conv.i = sext i32 %mul.i to i64, !dbg !204
  %b.i = getelementptr inbounds nuw i8, ptr %p, i64 8, !dbg !205
  %1 = load i64, ptr %b.i, align 8, !dbg !205, !tbaa !145
  %add.i = add i64 %1, 1, !dbg !206
  %add = add i64 %add.i, %conv.i, !dbg !207
  ret i64 %add, !dbg !208
}

define dso_local i64 @c13_after_split(double %m.coerce0, i64 %m.coerce1, i32 noundef %z) local_unnamed_addr !dbg !209 {
entry:
    #dbg_value(double %m.coerce0, !213, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !215)
    #dbg_value(i64 %m.coerce1, !213, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !215)
    #dbg_value(i32 %z, !214, !DIExpression(), !215)
  %conv = sext i32 %z to i64, !dbg !216
  %add = add nsw i64 %m.coerce1, %conv, !dbg !217
  ret i64 %add, !dbg !218
}

define dso_local noundef i64 @_Z8c11_crefRK4foo2(ptr nofree noundef nonnull readonly align 8 captures(none) dereferenceable(24) %r) local_unnamed_addr !dbg !219 {
entry:
    #dbg_value(ptr %r, !230, !DIExpression(), !231)
  %0 = load i32, ptr %r, align 8, !dbg !232, !tbaa !233
  %conv = sext i32 %0 to i64, !dbg !236
  %b = getelementptr inbounds nuw i8, ptr %r, i64 8, !dbg !237
  %1 = load i64, ptr %b, align 8, !dbg !237, !tbaa !238
  %add = add nsw i64 %1, %conv, !dbg !239
  ret i64 %add, !dbg !240
}

define dso_local noundef range(i64 -2147483648, 2147483648) i64 @_ZNK1C5scaleEi(ptr nofree noundef nonnull readonly align 4 captures(none) dereferenceable(4) %this, i32 noundef %f) local_unnamed_addr align 2 !dbg !241 {
entry:
    #dbg_value(ptr %this, !251, !DIExpression(), !254)
    #dbg_value(i32 %f, !253, !DIExpression(), !254)
  %0 = load i32, ptr %this, align 4, !dbg !255, !tbaa !256
  %mul = mul nsw i32 %0, %f, !dbg !258
  %conv = sext i32 %mul to i64, !dbg !259
  ret i64 %conv, !dbg !260
}

define dso_local noundef i64 @_Z14c14_nontrivial2NTi(ptr nofreeobj noundef readonly align 8 captures(none) dead_on_return dereferenceable(16) %n, i32 noundef %z) local_unnamed_addr !dbg !261 {
entry:
    #dbg_value(ptr %n, !266, !DIExpression(DW_OP_deref), !268)
    #dbg_value(i32 %z, !267, !DIExpression(), !268)
  %0 = load i64, ptr %n, align 8, !dbg !269, !tbaa !270
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %n, i64 8, !dbg !271
  %1 = load i64, ptr %arrayidx2, align 8, !dbg !271, !tbaa !270
  %conv = sext i32 %z to i64, !dbg !272
  %add = add i64 %0, %conv, !dbg !273
  %add3 = add i64 %add, %1, !dbg !274
  ret i64 %add3, !dbg !275
}

!llvm.dbg.cu = !{!0, !4}
!llvm.module.flags = !{!17, !18, !19, !20, !21, !22}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "cases.c", directory: "/")
!2 = !{!3}
!3 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !5, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "cases.cpp", directory: "/")
!7 = !{!8, !9, i64 0}
!8 = !{!"__libc_errno", !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !14, i64 0}
!13 = !{!"__libc_errno", !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 8, !"PIC Level", i32 2}
!20 = !{i32 7, !"PIE Level", i32 2}
!21 = !{i32 7, !"uwtable", i32 2}
!22 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!23 = distinct !DISubprogram(name: "c01_scalar", scope: !1, file: !1, line: 11, type: !24, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !27, keyInstructions: true)
!24 = !DISubroutineType(types: !25)
!25 = !{!3, !26, !3}
!26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "a", arg: 1, scope: !23, file: !1, line: 11, type: !26)
!29 = !DILocalVariable(name: "b", arg: 2, scope: !23, file: !1, line: 11, type: !3)
!30 = !DILocation(line: 0, scope: !23)
!31 = !DILocation(line: 11, column: 44, scope: !23)
!32 = !DILocation(line: 11, column: 46, scope: !23, atomGroup: 1, atomRank: 2)
!33 = !DILocation(line: 11, column: 37, scope: !23, atomGroup: 1, atomRank: 1)
!34 = distinct !DISubprogram(name: "c02_pointer", scope: !1, file: !1, line: 12, type: !35, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !39, keyInstructions: true)
!35 = !DISubroutineType(types: !36)
!36 = !{!3, !37}
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64)
!38 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!39 = !{!40}
!40 = !DILocalVariable(name: "p", arg: 1, scope: !34, file: !1, line: 12, type: !37)
!41 = !DILocation(line: 0, scope: !34)
!42 = !DILocation(line: 12, column: 45, scope: !34, atomGroup: 1, atomRank: 2)
!43 = !DILocation(line: 12, column: 38, scope: !34, atomGroup: 1, atomRank: 1)
!44 = distinct !DISubprogram(name: "c03_fp", scope: !1, file: !1, line: 13, type: !45, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !49, keyInstructions: true)
!45 = !DISubroutineType(types: !46)
!46 = !{!47, !48, !47}
!47 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!48 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!49 = !{!50, !51}
!50 = !DILocalVariable(name: "f", arg: 1, scope: !44, file: !1, line: 13, type: !48)
!51 = !DILocalVariable(name: "d", arg: 2, scope: !44, file: !1, line: 13, type: !47)
!52 = !DILocation(line: 0, scope: !44)
!53 = !DILocation(line: 13, column: 46, scope: !44)
!54 = !DILocation(line: 13, column: 48, scope: !44, atomGroup: 1, atomRank: 2)
!55 = !DILocation(line: 13, column: 39, scope: !44, atomGroup: 1, atomRank: 1)
!56 = distinct !DISubprogram(name: "c04_small", scope: !1, file: !1, line: 14, type: !57, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !63, keyInstructions: true)
!57 = !DISubroutineType(types: !58)
!58 = !{!26, !59}
!59 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !1, line: 6, size: 64, elements: !60)
!60 = !{!61, !62}
!61 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !59, file: !1, line: 6, baseType: !26, size: 32)
!62 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !59, file: !1, line: 6, baseType: !26, size: 32, offset: 32)
!63 = !{!64}
!64 = !DILocalVariable(name: "p", arg: 1, scope: !56, file: !1, line: 14, type: !59)
!65 = !DILocation(line: 0, scope: !56)
!66 = !DILocation(line: 14, column: 46, scope: !56, atomGroup: 1, atomRank: 2)
!67 = !DILocation(line: 14, column: 35, scope: !56, atomGroup: 1, atomRank: 1)
!68 = distinct !DISubprogram(name: "c05_split", scope: !1, file: !1, line: 15, type: !69, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !75, keyInstructions: true)
!69 = !DISubroutineType(types: !70)
!70 = !{!47, !71}
!71 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mix", file: !1, line: 7, size: 128, elements: !72)
!72 = !{!73, !74}
!73 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !71, file: !1, line: 7, baseType: !47, size: 64)
!74 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !71, file: !1, line: 7, baseType: !3, size: 64, offset: 64)
!75 = !{!76}
!76 = !DILocalVariable(name: "m", arg: 1, scope: !68, file: !1, line: 15, type: !71)
!77 = !DILocation(line: 0, scope: !68)
!78 = !DILocation(line: 15, column: 50, scope: !68)
!79 = !DILocation(line: 15, column: 48, scope: !68, atomGroup: 1, atomRank: 2)
!80 = !DILocation(line: 15, column: 37, scope: !68, atomGroup: 1, atomRank: 1)
!81 = distinct !DISubprogram(name: "c05_hfa", scope: !1, file: !1, line: 16, type: !82, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !89, keyInstructions: true)
!82 = !DISubroutineType(types: !83)
!83 = !{!48, !84}
!84 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "hfa", file: !1, line: 8, size: 96, elements: !85)
!85 = !{!86, !87, !88}
!86 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !84, file: !1, line: 8, baseType: !48, size: 32)
!87 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !84, file: !1, line: 8, baseType: !48, size: 32, offset: 32)
!88 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !84, file: !1, line: 8, baseType: !48, size: 32, offset: 64)
!89 = !{!90}
!90 = !DILocalVariable(name: "h", arg: 1, scope: !81, file: !1, line: 16, type: !84)
!91 = !DILocation(line: 0, scope: !81)
!92 = !DILocation(line: 16, column: 45, scope: !81)
!93 = !DILocation(line: 16, column: 51, scope: !81, atomGroup: 1, atomRank: 2)
!94 = !DILocation(line: 16, column: 34, scope: !81, atomGroup: 1, atomRank: 1)
!95 = distinct !DISubprogram(name: "c06_big", scope: !1, file: !1, line: 17, type: !96, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !104, keyInstructions: true)
!96 = !DISubroutineType(types: !97)
!97 = !{!3, !98}
!98 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "big", file: !1, line: 9, size: 256, elements: !99)
!99 = !{!100, !101, !102, !103}
!100 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !98, file: !1, line: 9, baseType: !3, size: 64)
!101 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !98, file: !1, line: 9, baseType: !3, size: 64, offset: 64)
!102 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !98, file: !1, line: 9, baseType: !3, size: 64, offset: 128)
!103 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !98, file: !1, line: 9, baseType: !3, size: 64, offset: 192)
!104 = !{!105}
!105 = !DILocalVariable(name: "b", arg: 1, scope: !95, file: !1, line: 17, type: !98)
!106 = !DILocation(line: 17, column: 28, scope: !95)
!107 = !DILocation(line: 17, column: 42, scope: !95)
!108 = !{!109, !110, i64 0}
!109 = !{!"big", !110, i64 0, !110, i64 8, !110, i64 16, !110, i64 24}
!110 = !{!"long", !10, i64 0}
!111 = !DILocation(line: 17, column: 48, scope: !95)
!112 = !{!109, !110, i64 24}
!113 = !DILocation(line: 17, column: 44, scope: !95, atomGroup: 1, atomRank: 2)
!114 = !DILocation(line: 17, column: 33, scope: !95, atomGroup: 1, atomRank: 1)
!115 = distinct !DISubprogram(name: "c07_sret", scope: !1, file: !1, line: 18, type: !116, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !118, keyInstructions: true)
!116 = !DISubroutineType(types: !117)
!117 = !{!98, !3}
!118 = !{!119, !120}
!119 = !DILocalVariable(name: "x", arg: 1, scope: !115, file: !1, line: 18, type: !3)
!120 = !DILocalVariable(name: "r", scope: !115, file: !1, line: 18, type: !98)
!121 = !DILocation(line: 0, scope: !115)
!122 = !DILocation(line: 18, column: 45, scope: !115)
!123 = !DILocation(line: 18, column: 55, scope: !115, atomGroup: 1, atomRank: 2)
!124 = !DILocation(line: 18, column: 49, scope: !115, atomGroup: 1, atomRank: 1)
!125 = !{!110, !110, i64 0}
!126 = !DILocation(line: 18, column: 75, scope: !115, atomGroup: 2, atomRank: 1)
!127 = distinct !DISubprogram(name: "c08_struct_ptr", scope: !1, file: !1, line: 19, type: !128, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !137, keyInstructions: true)
!128 = !DISubroutineType(types: !129)
!129 = !{!3, !130}
!130 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !131, size: 64)
!131 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 5, size: 192, elements: !132)
!132 = !{!133, !134, !135}
!133 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !131, file: !1, line: 5, baseType: !26, size: 32)
!134 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !131, file: !1, line: 5, baseType: !3, size: 64, offset: 64)
!135 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !131, file: !1, line: 5, baseType: !136, size: 8, offset: 128)
!136 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!137 = !{!138}
!138 = !DILocalVariable(name: "f", arg: 1, scope: !127, file: !1, line: 19, type: !130)
!139 = !DILocation(line: 0, scope: !127)
!140 = !DILocation(line: 19, column: 51, scope: !127)
!141 = !{!142, !9, i64 0}
!142 = !{!"foo", !9, i64 0, !110, i64 8, !10, i64 16}
!143 = !DILocation(line: 19, column: 48, scope: !127)
!144 = !DILocation(line: 19, column: 58, scope: !127)
!145 = !{!142, !110, i64 8}
!146 = !DILocation(line: 19, column: 53, scope: !127)
!147 = !DILocation(line: 19, column: 65, scope: !127)
!148 = !{!142, !10, i64 16}
!149 = !DILocation(line: 19, column: 62, scope: !127)
!150 = !DILocation(line: 19, column: 60, scope: !127, atomGroup: 1, atomRank: 2)
!151 = !DILocation(line: 19, column: 41, scope: !127, atomGroup: 1, atomRank: 1)
!152 = distinct !DISubprogram(name: "c09_stack", scope: !1, file: !1, line: 20, type: !153, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !155, keyInstructions: true)
!153 = !DISubroutineType(types: !154)
!154 = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!155 = !{!156, !157, !158, !159, !160, !161, !162, !163, !164, !165}
!156 = !DILocalVariable(name: "a0", arg: 1, scope: !152, file: !1, line: 20, type: !3)
!157 = !DILocalVariable(name: "a1", arg: 2, scope: !152, file: !1, line: 20, type: !3)
!158 = !DILocalVariable(name: "a2", arg: 3, scope: !152, file: !1, line: 20, type: !3)
!159 = !DILocalVariable(name: "a3", arg: 4, scope: !152, file: !1, line: 20, type: !3)
!160 = !DILocalVariable(name: "a4", arg: 5, scope: !152, file: !1, line: 20, type: !3)
!161 = !DILocalVariable(name: "a5", arg: 6, scope: !152, file: !1, line: 20, type: !3)
!162 = !DILocalVariable(name: "a6", arg: 7, scope: !152, file: !1, line: 21, type: !3)
!163 = !DILocalVariable(name: "a7", arg: 8, scope: !152, file: !1, line: 21, type: !3)
!164 = !DILocalVariable(name: "a8", arg: 9, scope: !152, file: !1, line: 21, type: !3)
!165 = !DILocalVariable(name: "a9", arg: 10, scope: !152, file: !1, line: 21, type: !3)
!166 = !DILocation(line: 0, scope: !152)
!167 = !DILocation(line: 22, column: 13, scope: !152)
!168 = !DILocation(line: 22, column: 18, scope: !152)
!169 = !DILocation(line: 22, column: 23, scope: !152)
!170 = !DILocation(line: 22, column: 28, scope: !152)
!171 = !DILocation(line: 22, column: 33, scope: !152)
!172 = !DILocation(line: 22, column: 38, scope: !152)
!173 = !DILocation(line: 22, column: 43, scope: !152)
!174 = !DILocation(line: 22, column: 48, scope: !152)
!175 = !DILocation(line: 22, column: 53, scope: !152, atomGroup: 1, atomRank: 2)
!176 = !DILocation(line: 22, column: 3, scope: !152, atomGroup: 1, atomRank: 1)
!177 = distinct !DISubprogram(name: "c10_mixed", scope: !1, file: !1, line: 24, type: !178, scopeLine: 24, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !180, keyInstructions: true)
!178 = !DISubroutineType(types: !179)
!179 = !{!47, !26, !47, !3, !48}
!180 = !{!181, !182, !183, !184}
!181 = !DILocalVariable(name: "i", arg: 1, scope: !177, file: !1, line: 24, type: !26)
!182 = !DILocalVariable(name: "d", arg: 2, scope: !177, file: !1, line: 24, type: !47)
!183 = !DILocalVariable(name: "l", arg: 3, scope: !177, file: !1, line: 24, type: !3)
!184 = !DILocalVariable(name: "f", arg: 4, scope: !177, file: !1, line: 24, type: !48)
!185 = !DILocation(line: 0, scope: !177)
!186 = !DILocation(line: 24, column: 64, scope: !177)
!187 = !DILocation(line: 24, column: 66, scope: !177)
!188 = !DILocation(line: 24, column: 72, scope: !177)
!189 = !DILocation(line: 24, column: 70, scope: !177)
!190 = !DILocation(line: 24, column: 76, scope: !177)
!191 = !DILocation(line: 24, column: 74, scope: !177, atomGroup: 1, atomRank: 2)
!192 = !DILocation(line: 24, column: 57, scope: !177, atomGroup: 1, atomRank: 1)
!193 = distinct !DISubprogram(name: "c12_outer", scope: !1, file: !1, line: 27, type: !128, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !194, keyInstructions: true)
!194 = !{!195}
!195 = !DILocalVariable(name: "p", arg: 1, scope: !193, file: !1, line: 27, type: !130)
!196 = !DILocation(line: 0, scope: !193)
!197 = !DILocalVariable(name: "p", arg: 1, scope: !198, file: !1, line: 26, type: !130)
!198 = distinct !DISubprogram(name: "inner", scope: !1, file: !1, line: 26, type: !128, scopeLine: 26, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !199, keyInstructions: true)
!199 = !{!197}
!200 = distinct !DILocation(line: 0, scope: !198, inlinedAt: !201)
!201 = distinct !DILocation(line: 27, column: 43, scope: !193)
!202 = distinct !DILocation(line: 26, column: 53, scope: !198, inlinedAt: !201)
!203 = distinct !DILocation(line: 26, column: 55, scope: !198, inlinedAt: !201)
!204 = distinct !DILocation(line: 26, column: 50, scope: !198, inlinedAt: !201)
!205 = distinct !DILocation(line: 26, column: 64, scope: !198, inlinedAt: !201)
!206 = distinct !DILocation(line: 26, column: 59, scope: !198, inlinedAt: !201, atomGroup: 1, atomRank: 2)
!207 = !DILocation(line: 27, column: 52, scope: !193, atomGroup: 1, atomRank: 2)
!208 = !DILocation(line: 27, column: 36, scope: !193, atomGroup: 1, atomRank: 1)
!209 = distinct !DISubprogram(name: "c13_after_split", scope: !1, file: !1, line: 30, type: !210, scopeLine: 30, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !212, keyInstructions: true)
!210 = !DISubroutineType(types: !211)
!211 = !{!3, !71, !26}
!212 = !{!213, !214}
!213 = !DILocalVariable(name: "m", arg: 1, scope: !209, file: !1, line: 30, type: !71)
!214 = !DILocalVariable(name: "z", arg: 2, scope: !209, file: !1, line: 30, type: !26)
!215 = !DILocation(line: 0, scope: !209)
!216 = !DILocation(line: 30, column: 61, scope: !209)
!217 = !DILocation(line: 30, column: 59, scope: !209, atomGroup: 1, atomRank: 2)
!218 = !DILocation(line: 30, column: 48, scope: !209, atomGroup: 1, atomRank: 1)
!219 = distinct !DISubprogram(name: "c11_cref", linkageName: "_Z8c11_crefRK4foo2", scope: !5, file: !5, line: 3, type: !220, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !229, keyInstructions: true)
!220 = !DISubroutineType(types: !221)
!221 = !{!3, !222}
!222 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !223, size: 64)
!223 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !224)
!224 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo2", file: !5, line: 2, size: 192, flags: DIFlagTypePassByValue, elements: !225, identifier: "_ZTS4foo2")
!225 = !{!226, !227, !228}
!226 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !224, file: !5, line: 2, baseType: !26, size: 32)
!227 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !224, file: !5, line: 2, baseType: !3, size: 64, offset: 64)
!228 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !224, file: !5, line: 2, baseType: !136, size: 8, offset: 128)
!229 = !{!230}
!230 = !DILocalVariable(name: "r", arg: 1, scope: !219, file: !5, line: 3, type: !222)
!231 = !DILocation(line: 0, scope: !219)
!232 = !DILocation(line: 3, column: 44, scope: !219)
!233 = !{!234, !14, i64 0}
!234 = !{!"_ZTS4foo2", !14, i64 0, !235, i64 8, !15, i64 16}
!235 = !{!"long", !15, i64 0}
!236 = !DILocation(line: 3, column: 42, scope: !219)
!237 = !DILocation(line: 3, column: 50, scope: !219)
!238 = !{!234, !235, i64 8}
!239 = !DILocation(line: 3, column: 46, scope: !219, atomGroup: 1, atomRank: 2)
!240 = !DILocation(line: 3, column: 35, scope: !219, atomGroup: 1, atomRank: 1)
!241 = distinct !DISubprogram(name: "scale", linkageName: "_ZNK1C5scaleEi", scope: !242, file: !5, line: 5, type: !246, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, declaration: !245, retainedNodes: !250, keyInstructions: true)
!242 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !5, line: 4, size: 32, flags: DIFlagTypePassByValue, elements: !243, identifier: "_ZTS1C")
!243 = !{!244, !245}
!244 = !DIDerivedType(tag: DW_TAG_member, name: "base", scope: !242, file: !5, line: 4, baseType: !26, size: 32)
!245 = !DISubprogram(name: "scale", linkageName: "_ZNK1C5scaleEi", scope: !242, file: !5, line: 4, type: !246, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!246 = !DISubroutineType(types: !247)
!247 = !{!3, !248, !26}
!248 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !249, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!249 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !242)
!250 = !{!251, !253}
!251 = !DILocalVariable(name: "this", arg: 1, scope: !241, type: !252, flags: DIFlagArtificial | DIFlagObjectPointer)
!252 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !249, size: 64)
!253 = !DILocalVariable(name: "f", arg: 2, scope: !241, file: !5, line: 5, type: !26)
!254 = !DILocation(line: 0, scope: !241)
!255 = !DILocation(line: 5, column: 40, scope: !241)
!256 = !{!257, !14, i64 0}
!257 = !{!"_ZTS1C", !14, i64 0}
!258 = !DILocation(line: 5, column: 45, scope: !241, atomGroup: 1, atomRank: 3)
!259 = !DILocation(line: 5, column: 40, scope: !241, atomGroup: 1, atomRank: 2)
!260 = !DILocation(line: 5, column: 33, scope: !241, atomGroup: 1, atomRank: 1)
!261 = distinct !DISubprogram(name: "c14_nontrivial", linkageName: "_Z14c14_nontrivial2NTi", scope: !5, file: !5, line: 9, type: !262, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !265, keyInstructions: true)
!262 = !DISubroutineType(types: !263)
!263 = !{!3, !264, !26}
!264 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NT", file: !5, line: 8, size: 128, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS2NT")
!265 = !{!266, !267}
!266 = !DILocalVariable(name: "n", arg: 1, scope: !261, file: !5, line: 9, type: !264)
!267 = !DILocalVariable(name: "z", arg: 2, scope: !261, file: !5, line: 9, type: !26)
!268 = !DILocation(line: 0, scope: !261)
!269 = !DILocation(line: 9, column: 46, scope: !261)
!270 = !{!235, !235, i64 0}
!271 = !DILocation(line: 9, column: 55, scope: !261)
!272 = !DILocation(line: 9, column: 64, scope: !261)
!273 = !DILocation(line: 9, column: 53, scope: !261)
!274 = !DILocation(line: 9, column: 62, scope: !261, atomGroup: 1, atomRank: 2)
!275 = !DILocation(line: 9, column: 39, scope: !261, atomGroup: 1, atomRank: 1)
