; A set of C and C++ argument shapes, as clang -O2 -g emits them
; for aarch64-linux-gnu: cases.c and cases.cpp, linked into one module.
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.big = type { i64, i64, i64, i64 }

define dso_local i64 @c01_scalar(i32 noundef %a, i64 noundef %b) local_unnamed_addr !dbg !30 {
entry:
    #dbg_value(i32 %a, !35, !DIExpression(), !37)
    #dbg_value(i64 %b, !36, !DIExpression(), !37)
  %conv = sext i32 %a to i64, !dbg !38
  %add = add nsw i64 %b, %conv, !dbg !39
  ret i64 %add, !dbg !40
}

define dso_local noundef i64 @c02_pointer(ptr noundef %p) local_unnamed_addr !dbg !41 {
entry:
    #dbg_value(ptr %p, !47, !DIExpression(), !48)
  %0 = ptrtoint ptr %p to i64, !dbg !49
  ret i64 %0, !dbg !50
}

define dso_local noundef double @c03_fp(float noundef %f, double noundef %d) local_unnamed_addr !dbg !51 {
entry:
    #dbg_value(float %f, !57, !DIExpression(), !59)
    #dbg_value(double %d, !58, !DIExpression(), !59)
  %conv = fpext float %f to double, !dbg !60
  %add = fadd double %d, %conv, !dbg !61
  ret double %add, !dbg !62
}

define dso_local i32 @c04_small(i64 %p.coerce) local_unnamed_addr !dbg !63 {
entry:
  %p.sroa.0.0.extract.trunc = trunc i64 %p.coerce to i32
  %p.sroa.2.0.extract.shift = lshr i64 %p.coerce, 32
  %p.sroa.2.0.extract.trunc = trunc nuw i64 %p.sroa.2.0.extract.shift to i32
    #dbg_value(i32 %p.sroa.0.0.extract.trunc, !71, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !72)
    #dbg_value(i32 %p.sroa.2.0.extract.trunc, !71, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !72)
  %add = add nsw i32 %p.sroa.2.0.extract.trunc, %p.sroa.0.0.extract.trunc, !dbg !73
  ret i32 %add, !dbg !74
}

define dso_local double @c05_split([2 x i64] %m.coerce) local_unnamed_addr !dbg !75 {
entry:
  %m.coerce.fca.0.extract = extractvalue [2 x i64] %m.coerce, 0
  %0 = bitcast i64 %m.coerce.fca.0.extract to double
  %m.coerce.fca.1.extract = extractvalue [2 x i64] %m.coerce, 1
    #dbg_value(double %0, !83, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !84)
    #dbg_value(i64 %m.coerce.fca.1.extract, !83, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !84)
  %conv = sitofp i64 %m.coerce.fca.1.extract to double, !dbg !85
  %add = fadd double %0, %conv, !dbg !86
  ret double %add, !dbg !87
}

define dso_local float @c05_hfa([3 x float] alignstack(8) %h.coerce) local_unnamed_addr !dbg !88 {
entry:
  %h.coerce.fca.0.extract = extractvalue [3 x float] %h.coerce, 0
  %h.coerce.fca.1.extract = extractvalue [3 x float] %h.coerce, 1
  %h.coerce.fca.2.extract = extractvalue [3 x float] %h.coerce, 2
    #dbg_value(float %h.coerce.fca.0.extract, !97, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !98)
    #dbg_value(float %h.coerce.fca.1.extract, !97, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !98)
    #dbg_value(float %h.coerce.fca.2.extract, !97, !DIExpression(DW_OP_LLVM_fragment, 64, 32), !98)
  %add = fadd float %h.coerce.fca.0.extract, %h.coerce.fca.1.extract, !dbg !99
  %add1 = fadd float %h.coerce.fca.2.extract, %add, !dbg !100
  ret float %add1, !dbg !101
}

define dso_local i64 @c06_big(ptr nofreeobj noundef readonly align 8 captures(none) dead_on_return dereferenceable(32) %b) local_unnamed_addr !dbg !102 {
entry:
    #dbg_value(ptr %b, !112, !DIExpression(DW_OP_deref), !113)
  %0 = load i64, ptr %b, align 8, !dbg !114, !tbaa !115
  %d = getelementptr inbounds nuw i8, ptr %b, i64 24, !dbg !118
  %1 = load i64, ptr %d, align 8, !dbg !118, !tbaa !119
  %add = add nsw i64 %1, %0, !dbg !120
  ret i64 %add, !dbg !121
}

define dso_local void @c07_sret(ptr dead_on_unwind noalias nofree writable writeonly sret(%struct.big) align 8 captures(none) initializes((0, 32)) %agg.result, i64 noundef %x) local_unnamed_addr !dbg !122 {
entry:
    #dbg_value(i64 %x, !126, !DIExpression(), !128)
    #dbg_declare(ptr %agg.result, !127, !DIExpression(), !129)
  store i64 %x, ptr %agg.result, align 8, !dbg !130, !tbaa !115
  %b = getelementptr inbounds nuw i8, ptr %agg.result, i64 8, !dbg !131
  %add = add nsw i64 %x, 1, !dbg !132
  store i64 %add, ptr %b, align 8, !dbg !130, !tbaa !133
  %c = getelementptr inbounds nuw i8, ptr %agg.result, i64 16, !dbg !131
  %add1 = add nsw i64 %x, 2, !dbg !134
  store i64 %add1, ptr %c, align 8, !dbg !130, !tbaa !135
  %d = getelementptr inbounds nuw i8, ptr %agg.result, i64 24, !dbg !131
  %add2 = add nsw i64 %x, 3, !dbg !136
  store i64 %add2, ptr %d, align 8, !dbg !130, !tbaa !119
  ret void, !dbg !137
}

define dso_local i64 @c08_struct_ptr(ptr nofree noundef readonly captures(none) %f) local_unnamed_addr !dbg !138 {
entry:
    #dbg_value(ptr %f, !149, !DIExpression(), !150)
  %0 = load i32, ptr %f, align 8, !dbg !151, !tbaa !152
  %conv = sext i32 %0 to i64, !dbg !154
  %b = getelementptr inbounds nuw i8, ptr %f, i64 8, !dbg !155
  %1 = load i64, ptr %b, align 8, !dbg !155, !tbaa !156
  %add = add nsw i64 %1, %conv, !dbg !157
  %c = getelementptr inbounds nuw i8, ptr %f, i64 16, !dbg !158
  %2 = load i8, ptr %c, align 8, !dbg !158, !tbaa !159
  %conv1 = zext i8 %2 to i64, !dbg !160
  %add2 = add nsw i64 %add, %conv1, !dbg !161
  ret i64 %add2, !dbg !162
}

define dso_local i64 @c09_stack(i64 noundef %a0, i64 noundef %a1, i64 noundef %a2, i64 noundef %a3, i64 noundef %a4, i64 noundef %a5, i64 noundef %a6, i64 noundef %a7, i64 noundef %a8, i64 noundef %a9) local_unnamed_addr !dbg !163 {
entry:
    #dbg_value(i64 %a0, !167, !DIExpression(), !177)
    #dbg_value(i64 %a1, !168, !DIExpression(), !177)
    #dbg_value(i64 %a2, !169, !DIExpression(), !177)
    #dbg_value(i64 %a3, !170, !DIExpression(), !177)
    #dbg_value(i64 %a4, !171, !DIExpression(), !177)
    #dbg_value(i64 %a5, !172, !DIExpression(), !177)
    #dbg_value(i64 %a6, !173, !DIExpression(), !177)
    #dbg_value(i64 %a7, !174, !DIExpression(), !177)
    #dbg_value(i64 %a8, !175, !DIExpression(), !177)
    #dbg_value(i64 %a9, !176, !DIExpression(), !177)
  %add = add nsw i64 %a1, %a0, !dbg !178
  %add1 = add nsw i64 %add, %a2, !dbg !179
  %add2 = add nsw i64 %add1, %a3, !dbg !180
  %add3 = add nsw i64 %add2, %a4, !dbg !181
  %add4 = add nsw i64 %add3, %a5, !dbg !182
  %add5 = add nsw i64 %add4, %a6, !dbg !183
  %add6 = add nsw i64 %add5, %a7, !dbg !184
  %add7 = add nsw i64 %add6, %a8, !dbg !185
  %add8 = add nsw i64 %add7, %a9, !dbg !186
  ret i64 %add8, !dbg !187
}

define dso_local noundef double @c10_mixed(i32 noundef %i, double noundef %d, i64 noundef %l, float noundef %f) local_unnamed_addr !dbg !188 {
entry:
    #dbg_value(i32 %i, !192, !DIExpression(), !196)
    #dbg_value(double %d, !193, !DIExpression(), !196)
    #dbg_value(i64 %l, !194, !DIExpression(), !196)
    #dbg_value(float %f, !195, !DIExpression(), !196)
  %conv = sitofp i32 %i to double, !dbg !197
  %add = fadd double %d, %conv, !dbg !198
  %conv1 = sitofp i64 %l to double, !dbg !199
  %add2 = fadd double %add, %conv1, !dbg !200
  %conv3 = fpext float %f to double, !dbg !201
  %add4 = fadd double %add2, %conv3, !dbg !202
  ret double %add4, !dbg !203
}

define dso_local range(i64 -9223372036854775807, -9223372036854775808) i64 @c12_outer(ptr nofree noundef readonly captures(none) %p) local_unnamed_addr !dbg !204 {
entry:
    #dbg_value(ptr %p, !206, !DIExpression(), !207)
    #dbg_value(ptr %p, !208, !DIExpression(), !211)
  %0 = load i32, ptr %p, align 8, !dbg !213, !tbaa !152
  %mul.i = mul nsw i32 %0, 3, !dbg !214
  %conv.i = sext i32 %mul.i to i64, !dbg !215
  %b.i = getelementptr inbounds nuw i8, ptr %p, i64 8, !dbg !216
  %1 = load i64, ptr %b.i, align 8, !dbg !216, !tbaa !156
  %add.i = add i64 %1, 1, !dbg !217
  %add = add i64 %add.i, %conv.i, !dbg !218
  ret i64 %add, !dbg !219
}

define dso_local i64 @c13_after_split([2 x i64] %m.coerce, i32 noundef %z) local_unnamed_addr !dbg !220 {
entry:
  %m.coerce.fca.1.extract = extractvalue [2 x i64] %m.coerce, 1
    #dbg_value(i64 poison, !224, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !226)
    #dbg_value(i64 %m.coerce.fca.1.extract, !224, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !226)
    #dbg_value(i32 %z, !225, !DIExpression(), !226)
  %conv = sext i32 %z to i64, !dbg !227
  %add = add nsw i64 %m.coerce.fca.1.extract, %conv, !dbg !228
  ret i64 %add, !dbg !229
}

define dso_local noundef i64 @_Z8c11_crefRK4foo2(ptr nofree noundef nonnull readonly align 8 captures(none) dereferenceable(24) %r) local_unnamed_addr !dbg !230 {
entry:
    #dbg_value(ptr %r, !241, !DIExpression(), !242)
  %0 = load i32, ptr %r, align 8, !dbg !243, !tbaa !244
  %conv = sext i32 %0 to i64, !dbg !247
  %b = getelementptr inbounds nuw i8, ptr %r, i64 8, !dbg !248
  %1 = load i64, ptr %b, align 8, !dbg !248, !tbaa !249
  %add = add nsw i64 %1, %conv, !dbg !250
  ret i64 %add, !dbg !251
}

define dso_local noundef range(i64 -2147483648, 2147483648) i64 @_ZNK1C5scaleEi(ptr nofree noundef nonnull readonly align 4 captures(none) dereferenceable(4) %this, i32 noundef %f) local_unnamed_addr !dbg !252 {
entry:
    #dbg_value(ptr %this, !262, !DIExpression(), !265)
    #dbg_value(i32 %f, !264, !DIExpression(), !265)
  %0 = load i32, ptr %this, align 4, !dbg !266, !tbaa !267
  %mul = mul nsw i32 %0, %f, !dbg !269
  %conv = sext i32 %mul to i64, !dbg !270
  ret i64 %conv, !dbg !271
}

define dso_local noundef i64 @_Z14c14_nontrivial2NTi(ptr nofreeobj noundef readonly align 8 captures(none) dead_on_return dereferenceable(16) %n, i32 noundef %z) local_unnamed_addr !dbg !272 {
entry:
    #dbg_value(ptr %n, !277, !DIExpression(DW_OP_deref), !279)
    #dbg_value(i32 %z, !278, !DIExpression(), !279)
  %0 = load i64, ptr %n, align 8, !dbg !280, !tbaa !281
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %n, i64 8, !dbg !282
  %1 = load i64, ptr %arrayidx2, align 8, !dbg !282, !tbaa !281
  %conv = sext i32 %z to i64, !dbg !283
  %add = add i64 %0, %conv, !dbg !284
  %add3 = add i64 %add, %1, !dbg !285
  ret i64 %add3, !dbg !286
}

!llvm.dbg.cu = !{!0, !4}
!llvm.module.flags = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}

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
!19 = !{i32 1, !"ptrauth-elf-got", i32 0}
!20 = !{i32 1, !"ptrauth-init-fini", i32 0}
!21 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}
!22 = !{i32 1, !"ptrauth-sign-personality", i32 0}
!23 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!24 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 0}
!25 = !{i32 8, !"PIC Level", i32 2}
!26 = !{i32 7, !"PIE Level", i32 2}
!27 = !{i32 7, !"uwtable", i32 2}
!28 = !{i32 7, !"frame-pointer", i32 4}
!29 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!30 = distinct !DISubprogram(name: "c01_scalar", scope: !1, file: !1, line: 11, type: !31, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !34, keyInstructions: true)
!31 = !DISubroutineType(types: !32)
!32 = !{!3, !33, !3}
!33 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!34 = !{!35, !36}
!35 = !DILocalVariable(name: "a", arg: 1, scope: !30, file: !1, line: 11, type: !33)
!36 = !DILocalVariable(name: "b", arg: 2, scope: !30, file: !1, line: 11, type: !3)
!37 = !DILocation(line: 0, scope: !30)
!38 = !DILocation(line: 11, column: 44, scope: !30)
!39 = !DILocation(line: 11, column: 46, scope: !30, atomGroup: 1, atomRank: 2)
!40 = !DILocation(line: 11, column: 37, scope: !30, atomGroup: 1, atomRank: 1)
!41 = distinct !DISubprogram(name: "c02_pointer", scope: !1, file: !1, line: 12, type: !42, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46, keyInstructions: true)
!42 = !DISubroutineType(types: !43)
!43 = !{!3, !44}
!44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !45, size: 64)
!45 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!46 = !{!47}
!47 = !DILocalVariable(name: "p", arg: 1, scope: !41, file: !1, line: 12, type: !44)
!48 = !DILocation(line: 0, scope: !41)
!49 = !DILocation(line: 12, column: 45, scope: !41, atomGroup: 1, atomRank: 2)
!50 = !DILocation(line: 12, column: 38, scope: !41, atomGroup: 1, atomRank: 1)
!51 = distinct !DISubprogram(name: "c03_fp", scope: !1, file: !1, line: 13, type: !52, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !56, keyInstructions: true)
!52 = !DISubroutineType(types: !53)
!53 = !{!54, !55, !54}
!54 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!55 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!56 = !{!57, !58}
!57 = !DILocalVariable(name: "f", arg: 1, scope: !51, file: !1, line: 13, type: !55)
!58 = !DILocalVariable(name: "d", arg: 2, scope: !51, file: !1, line: 13, type: !54)
!59 = !DILocation(line: 0, scope: !51)
!60 = !DILocation(line: 13, column: 46, scope: !51)
!61 = !DILocation(line: 13, column: 48, scope: !51, atomGroup: 1, atomRank: 2)
!62 = !DILocation(line: 13, column: 39, scope: !51, atomGroup: 1, atomRank: 1)
!63 = distinct !DISubprogram(name: "c04_small", scope: !1, file: !1, line: 14, type: !64, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !70, keyInstructions: true)
!64 = !DISubroutineType(types: !65)
!65 = !{!33, !66}
!66 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !1, line: 6, size: 64, elements: !67)
!67 = !{!68, !69}
!68 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !66, file: !1, line: 6, baseType: !33, size: 32)
!69 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !66, file: !1, line: 6, baseType: !33, size: 32, offset: 32)
!70 = !{!71}
!71 = !DILocalVariable(name: "p", arg: 1, scope: !63, file: !1, line: 14, type: !66)
!72 = !DILocation(line: 0, scope: !63)
!73 = !DILocation(line: 14, column: 46, scope: !63, atomGroup: 1, atomRank: 2)
!74 = !DILocation(line: 14, column: 35, scope: !63, atomGroup: 1, atomRank: 1)
!75 = distinct !DISubprogram(name: "c05_split", scope: !1, file: !1, line: 15, type: !76, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !82, keyInstructions: true)
!76 = !DISubroutineType(types: !77)
!77 = !{!54, !78}
!78 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mix", file: !1, line: 7, size: 128, elements: !79)
!79 = !{!80, !81}
!80 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !78, file: !1, line: 7, baseType: !54, size: 64)
!81 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !78, file: !1, line: 7, baseType: !3, size: 64, offset: 64)
!82 = !{!83}
!83 = !DILocalVariable(name: "m", arg: 1, scope: !75, file: !1, line: 15, type: !78)
!84 = !DILocation(line: 0, scope: !75)
!85 = !DILocation(line: 15, column: 50, scope: !75)
!86 = !DILocation(line: 15, column: 48, scope: !75, atomGroup: 1, atomRank: 2)
!87 = !DILocation(line: 15, column: 37, scope: !75, atomGroup: 1, atomRank: 1)
!88 = distinct !DISubprogram(name: "c05_hfa", scope: !1, file: !1, line: 16, type: !89, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !96, keyInstructions: true)
!89 = !DISubroutineType(types: !90)
!90 = !{!55, !91}
!91 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "hfa", file: !1, line: 8, size: 96, elements: !92)
!92 = !{!93, !94, !95}
!93 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !91, file: !1, line: 8, baseType: !55, size: 32)
!94 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !91, file: !1, line: 8, baseType: !55, size: 32, offset: 32)
!95 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !91, file: !1, line: 8, baseType: !55, size: 32, offset: 64)
!96 = !{!97}
!97 = !DILocalVariable(name: "h", arg: 1, scope: !88, file: !1, line: 16, type: !91)
!98 = !DILocation(line: 0, scope: !88)
!99 = !DILocation(line: 16, column: 45, scope: !88)
!100 = !DILocation(line: 16, column: 51, scope: !88, atomGroup: 1, atomRank: 2)
!101 = !DILocation(line: 16, column: 34, scope: !88, atomGroup: 1, atomRank: 1)
!102 = distinct !DISubprogram(name: "c06_big", scope: !1, file: !1, line: 17, type: !103, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !111, keyInstructions: true)
!103 = !DISubroutineType(types: !104)
!104 = !{!3, !105}
!105 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "big", file: !1, line: 9, size: 256, elements: !106)
!106 = !{!107, !108, !109, !110}
!107 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !105, file: !1, line: 9, baseType: !3, size: 64)
!108 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !105, file: !1, line: 9, baseType: !3, size: 64, offset: 64)
!109 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !105, file: !1, line: 9, baseType: !3, size: 64, offset: 128)
!110 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !105, file: !1, line: 9, baseType: !3, size: 64, offset: 192)
!111 = !{!112}
!112 = !DILocalVariable(name: "b", arg: 1, scope: !102, file: !1, line: 17, type: !105)
!113 = !DILocation(line: 0, scope: !102)
!114 = !DILocation(line: 17, column: 42, scope: !102)
!115 = !{!116, !117, i64 0}
!116 = !{!"big", !117, i64 0, !117, i64 8, !117, i64 16, !117, i64 24}
!117 = !{!"long", !10, i64 0}
!118 = !DILocation(line: 17, column: 48, scope: !102)
!119 = !{!116, !117, i64 24}
!120 = !DILocation(line: 17, column: 44, scope: !102, atomGroup: 1, atomRank: 2)
!121 = !DILocation(line: 17, column: 33, scope: !102, atomGroup: 1, atomRank: 1)
!122 = distinct !DISubprogram(name: "c07_sret", scope: !1, file: !1, line: 18, type: !123, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !125, keyInstructions: true)
!123 = !DISubroutineType(types: !124)
!124 = !{!105, !3}
!125 = !{!126, !127}
!126 = !DILocalVariable(name: "x", arg: 1, scope: !122, file: !1, line: 18, type: !3)
!127 = !DILocalVariable(name: "r", scope: !122, file: !1, line: 18, type: !105)
!128 = !DILocation(line: 0, scope: !122)
!129 = !DILocation(line: 18, column: 45, scope: !122)
!130 = !DILocation(line: 18, column: 49, scope: !122, atomGroup: 1, atomRank: 1)
!131 = !DILocation(line: 18, column: 49, scope: !122)
!132 = !DILocation(line: 18, column: 55, scope: !122, atomGroup: 1, atomRank: 2)
!133 = !{!116, !117, i64 8}
!134 = !DILocation(line: 18, column: 62, scope: !122, atomGroup: 1, atomRank: 2)
!135 = !{!116, !117, i64 16}
!136 = !DILocation(line: 18, column: 69, scope: !122, atomGroup: 1, atomRank: 2)
!137 = !DILocation(line: 18, column: 75, scope: !122, atomGroup: 2, atomRank: 1)
!138 = distinct !DISubprogram(name: "c08_struct_ptr", scope: !1, file: !1, line: 19, type: !139, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !148, keyInstructions: true)
!139 = !DISubroutineType(types: !140)
!140 = !{!3, !141}
!141 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !142, size: 64)
!142 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 5, size: 192, elements: !143)
!143 = !{!144, !145, !146}
!144 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !142, file: !1, line: 5, baseType: !33, size: 32)
!145 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !142, file: !1, line: 5, baseType: !3, size: 64, offset: 64)
!146 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !142, file: !1, line: 5, baseType: !147, size: 8, offset: 128)
!147 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!148 = !{!149}
!149 = !DILocalVariable(name: "f", arg: 1, scope: !138, file: !1, line: 19, type: !141)
!150 = !DILocation(line: 0, scope: !138)
!151 = !DILocation(line: 19, column: 51, scope: !138)
!152 = !{!153, !9, i64 0}
!153 = !{!"foo", !9, i64 0, !117, i64 8, !10, i64 16}
!154 = !DILocation(line: 19, column: 48, scope: !138)
!155 = !DILocation(line: 19, column: 58, scope: !138)
!156 = !{!153, !117, i64 8}
!157 = !DILocation(line: 19, column: 53, scope: !138)
!158 = !DILocation(line: 19, column: 65, scope: !138)
!159 = !{!153, !10, i64 16}
!160 = !DILocation(line: 19, column: 62, scope: !138)
!161 = !DILocation(line: 19, column: 60, scope: !138, atomGroup: 1, atomRank: 2)
!162 = !DILocation(line: 19, column: 41, scope: !138, atomGroup: 1, atomRank: 1)
!163 = distinct !DISubprogram(name: "c09_stack", scope: !1, file: !1, line: 20, type: !164, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !166, keyInstructions: true)
!164 = !DISubroutineType(types: !165)
!165 = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!166 = !{!167, !168, !169, !170, !171, !172, !173, !174, !175, !176}
!167 = !DILocalVariable(name: "a0", arg: 1, scope: !163, file: !1, line: 20, type: !3)
!168 = !DILocalVariable(name: "a1", arg: 2, scope: !163, file: !1, line: 20, type: !3)
!169 = !DILocalVariable(name: "a2", arg: 3, scope: !163, file: !1, line: 20, type: !3)
!170 = !DILocalVariable(name: "a3", arg: 4, scope: !163, file: !1, line: 20, type: !3)
!171 = !DILocalVariable(name: "a4", arg: 5, scope: !163, file: !1, line: 20, type: !3)
!172 = !DILocalVariable(name: "a5", arg: 6, scope: !163, file: !1, line: 20, type: !3)
!173 = !DILocalVariable(name: "a6", arg: 7, scope: !163, file: !1, line: 21, type: !3)
!174 = !DILocalVariable(name: "a7", arg: 8, scope: !163, file: !1, line: 21, type: !3)
!175 = !DILocalVariable(name: "a8", arg: 9, scope: !163, file: !1, line: 21, type: !3)
!176 = !DILocalVariable(name: "a9", arg: 10, scope: !163, file: !1, line: 21, type: !3)
!177 = !DILocation(line: 0, scope: !163)
!178 = !DILocation(line: 22, column: 13, scope: !163)
!179 = !DILocation(line: 22, column: 18, scope: !163)
!180 = !DILocation(line: 22, column: 23, scope: !163)
!181 = !DILocation(line: 22, column: 28, scope: !163)
!182 = !DILocation(line: 22, column: 33, scope: !163)
!183 = !DILocation(line: 22, column: 38, scope: !163)
!184 = !DILocation(line: 22, column: 43, scope: !163)
!185 = !DILocation(line: 22, column: 48, scope: !163)
!186 = !DILocation(line: 22, column: 53, scope: !163, atomGroup: 1, atomRank: 2)
!187 = !DILocation(line: 22, column: 3, scope: !163, atomGroup: 1, atomRank: 1)
!188 = distinct !DISubprogram(name: "c10_mixed", scope: !1, file: !1, line: 24, type: !189, scopeLine: 24, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !191, keyInstructions: true)
!189 = !DISubroutineType(types: !190)
!190 = !{!54, !33, !54, !3, !55}
!191 = !{!192, !193, !194, !195}
!192 = !DILocalVariable(name: "i", arg: 1, scope: !188, file: !1, line: 24, type: !33)
!193 = !DILocalVariable(name: "d", arg: 2, scope: !188, file: !1, line: 24, type: !54)
!194 = !DILocalVariable(name: "l", arg: 3, scope: !188, file: !1, line: 24, type: !3)
!195 = !DILocalVariable(name: "f", arg: 4, scope: !188, file: !1, line: 24, type: !55)
!196 = !DILocation(line: 0, scope: !188)
!197 = !DILocation(line: 24, column: 64, scope: !188)
!198 = !DILocation(line: 24, column: 66, scope: !188)
!199 = !DILocation(line: 24, column: 72, scope: !188)
!200 = !DILocation(line: 24, column: 70, scope: !188)
!201 = !DILocation(line: 24, column: 76, scope: !188)
!202 = !DILocation(line: 24, column: 74, scope: !188, atomGroup: 1, atomRank: 2)
!203 = !DILocation(line: 24, column: 57, scope: !188, atomGroup: 1, atomRank: 1)
!204 = distinct !DISubprogram(name: "c12_outer", scope: !1, file: !1, line: 27, type: !139, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !205, keyInstructions: true)
!205 = !{!206}
!206 = !DILocalVariable(name: "p", arg: 1, scope: !204, file: !1, line: 27, type: !141)
!207 = !DILocation(line: 0, scope: !204)
!208 = !DILocalVariable(name: "p", arg: 1, scope: !209, file: !1, line: 26, type: !141)
!209 = distinct !DISubprogram(name: "inner", scope: !1, file: !1, line: 26, type: !139, scopeLine: 26, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !210, keyInstructions: true)
!210 = !{!208}
!211 = distinct !DILocation(line: 0, scope: !209, inlinedAt: !212)
!212 = distinct !DILocation(line: 27, column: 43, scope: !204)
!213 = distinct !DILocation(line: 26, column: 53, scope: !209, inlinedAt: !212)
!214 = distinct !DILocation(line: 26, column: 55, scope: !209, inlinedAt: !212)
!215 = distinct !DILocation(line: 26, column: 50, scope: !209, inlinedAt: !212)
!216 = distinct !DILocation(line: 26, column: 64, scope: !209, inlinedAt: !212)
!217 = distinct !DILocation(line: 26, column: 59, scope: !209, inlinedAt: !212, atomGroup: 1, atomRank: 2)
!218 = !DILocation(line: 27, column: 52, scope: !204, atomGroup: 1, atomRank: 2)
!219 = !DILocation(line: 27, column: 36, scope: !204, atomGroup: 1, atomRank: 1)
!220 = distinct !DISubprogram(name: "c13_after_split", scope: !1, file: !1, line: 30, type: !221, scopeLine: 30, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !223, keyInstructions: true)
!221 = !DISubroutineType(types: !222)
!222 = !{!3, !78, !33}
!223 = !{!224, !225}
!224 = !DILocalVariable(name: "m", arg: 1, scope: !220, file: !1, line: 30, type: !78)
!225 = !DILocalVariable(name: "z", arg: 2, scope: !220, file: !1, line: 30, type: !33)
!226 = !DILocation(line: 0, scope: !220)
!227 = !DILocation(line: 30, column: 61, scope: !220)
!228 = !DILocation(line: 30, column: 59, scope: !220, atomGroup: 1, atomRank: 2)
!229 = !DILocation(line: 30, column: 48, scope: !220, atomGroup: 1, atomRank: 1)
!230 = distinct !DISubprogram(name: "c11_cref", linkageName: "_Z8c11_crefRK4foo2", scope: !5, file: !5, line: 3, type: !231, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !240, keyInstructions: true)
!231 = !DISubroutineType(types: !232)
!232 = !{!3, !233}
!233 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !234, size: 64)
!234 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !235)
!235 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo2", file: !5, line: 2, size: 192, flags: DIFlagTypePassByValue, elements: !236, identifier: "_ZTS4foo2")
!236 = !{!237, !238, !239}
!237 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !235, file: !5, line: 2, baseType: !33, size: 32)
!238 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !235, file: !5, line: 2, baseType: !3, size: 64, offset: 64)
!239 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !235, file: !5, line: 2, baseType: !147, size: 8, offset: 128)
!240 = !{!241}
!241 = !DILocalVariable(name: "r", arg: 1, scope: !230, file: !5, line: 3, type: !233)
!242 = !DILocation(line: 0, scope: !230)
!243 = !DILocation(line: 3, column: 44, scope: !230)
!244 = !{!245, !14, i64 0}
!245 = !{!"_ZTS4foo2", !14, i64 0, !246, i64 8, !15, i64 16}
!246 = !{!"long", !15, i64 0}
!247 = !DILocation(line: 3, column: 42, scope: !230)
!248 = !DILocation(line: 3, column: 50, scope: !230)
!249 = !{!245, !246, i64 8}
!250 = !DILocation(line: 3, column: 46, scope: !230, atomGroup: 1, atomRank: 2)
!251 = !DILocation(line: 3, column: 35, scope: !230, atomGroup: 1, atomRank: 1)
!252 = distinct !DISubprogram(name: "scale", linkageName: "_ZNK1C5scaleEi", scope: !253, file: !5, line: 5, type: !257, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, declaration: !256, retainedNodes: !261, keyInstructions: true)
!253 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !5, line: 4, size: 32, flags: DIFlagTypePassByValue, elements: !254, identifier: "_ZTS1C")
!254 = !{!255, !256}
!255 = !DIDerivedType(tag: DW_TAG_member, name: "base", scope: !253, file: !5, line: 4, baseType: !33, size: 32)
!256 = !DISubprogram(name: "scale", linkageName: "_ZNK1C5scaleEi", scope: !253, file: !5, line: 4, type: !257, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!257 = !DISubroutineType(types: !258)
!258 = !{!3, !259, !33}
!259 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !260, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!260 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !253)
!261 = !{!262, !264}
!262 = !DILocalVariable(name: "this", arg: 1, scope: !252, type: !263, flags: DIFlagArtificial | DIFlagObjectPointer)
!263 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !260, size: 64)
!264 = !DILocalVariable(name: "f", arg: 2, scope: !252, file: !5, line: 5, type: !33)
!265 = !DILocation(line: 0, scope: !252)
!266 = !DILocation(line: 5, column: 40, scope: !252)
!267 = !{!268, !14, i64 0}
!268 = !{!"_ZTS1C", !14, i64 0}
!269 = !DILocation(line: 5, column: 45, scope: !252, atomGroup: 1, atomRank: 3)
!270 = !DILocation(line: 5, column: 40, scope: !252, atomGroup: 1, atomRank: 2)
!271 = !DILocation(line: 5, column: 33, scope: !252, atomGroup: 1, atomRank: 1)
!272 = distinct !DISubprogram(name: "c14_nontrivial", linkageName: "_Z14c14_nontrivial2NTi", scope: !5, file: !5, line: 9, type: !273, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !276, keyInstructions: true)
!273 = !DISubroutineType(types: !274)
!274 = !{!3, !275, !33}
!275 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NT", file: !5, line: 8, size: 128, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS2NT")
!276 = !{!277, !278}
!277 = !DILocalVariable(name: "n", arg: 1, scope: !272, file: !5, line: 9, type: !275)
!278 = !DILocalVariable(name: "z", arg: 2, scope: !272, file: !5, line: 9, type: !33)
!279 = !DILocation(line: 0, scope: !272)
!280 = !DILocation(line: 9, column: 46, scope: !272)
!281 = !{!246, !246, i64 0}
!282 = !DILocation(line: 9, column: 55, scope: !272)
!283 = !DILocation(line: 9, column: 64, scope: !272)
!284 = !DILocation(line: 9, column: 53, scope: !272)
!285 = !DILocation(line: 9, column: 62, scope: !272, atomGroup: 1, atomRank: 2)
!286 = !DILocation(line: 9, column: 39, scope: !272, atomGroup: 1, atomRank: 1)
