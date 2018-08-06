; RUN: llc %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -enable-misched
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; rdar://13183203
; Make sure when misched is enabled, we still have location information for
; function parameters.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "Proc8"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "Array1Par"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "Array2Par"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "IntParI1"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "IntParI2"

source_filename = "test/DebugInfo/X86/misched-dbg-value.ll"

%struct.Record = type { %struct.Record*, i32, i32, i32, [31 x i8] }

@Version = global [4 x i8] c"1.1\00", align 1, !dbg !0
@IntGlob = common global i32 0, align 4, !dbg !7
@BoolGlob = common global i32 0, align 4, !dbg !10
@Char1Glob = common global i8 0, align 1, !dbg !13
@Char2Glob = common global i8 0, align 1, !dbg !15
@Array1Glob = common global [51 x i32] zeroinitializer, align 16, !dbg !17
@Array2Glob = common global [51 x [51 x i32]] zeroinitializer, align 16, !dbg !23
@PtrGlb = common global %struct.Record* null, align 8, !dbg !28
@PtrGlbNext = common global %struct.Record* null, align 8, !dbg !54

; Function Attrs: nounwind optsize
define void @Proc8(i32* nocapture %Array1Par, [51 x i32]* nocapture %Array2Par, i32 %IntParI1, i32 %IntParI2) #0 !dbg !61 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %Array1Par, metadata !67, metadata !73), !dbg !74
  tail call void @llvm.dbg.value(metadata [51 x i32]* %Array2Par, metadata !68, metadata !73), !dbg !75
  tail call void @llvm.dbg.value(metadata i32 %IntParI1, metadata !69, metadata !73), !dbg !76
  tail call void @llvm.dbg.value(metadata i32 %IntParI2, metadata !70, metadata !73), !dbg !77
  %add = add i32 %IntParI1, 5, !dbg !78
  tail call void @llvm.dbg.value(metadata i32 %add, metadata !71, metadata !73), !dbg !78
  %idxprom = sext i32 %add to i64, !dbg !79
  %arrayidx = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom, !dbg !79
  store i32 %IntParI2, i32* %arrayidx, align 4, !dbg !79
  %add3 = add nsw i32 %IntParI1, 6, !dbg !80
  %idxprom4 = sext i32 %add3 to i64, !dbg !80
  %arrayidx5 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom4, !dbg !80
  store i32 %IntParI2, i32* %arrayidx5, align 4, !dbg !80
  %add6 = add nsw i32 %IntParI1, 35, !dbg !81
  %idxprom7 = sext i32 %add6 to i64, !dbg !81
  %arrayidx8 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom7, !dbg !81
  store i32 %add, i32* %arrayidx8, align 4, !dbg !81
  tail call void @llvm.dbg.value(metadata i32 %add, metadata !72, metadata !73), !dbg !82
  br label %for.body, !dbg !82

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %idxprom, %entry ], [ %indvars.iv.next, %for.body ]
  %IntIndex.046 = phi i32 [ %add, %entry ], [ %inc, %for.body ]
  %arrayidx13 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %indvars.iv, !dbg !84
  store i32 %add, i32* %arrayidx13, align 4, !dbg !84
  %inc = add nsw i32 %IntIndex.046, 1, !dbg !82
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !72, metadata !73), !dbg !82
  %cmp = icmp sgt i32 %inc, %add3, !dbg !82
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !82
  br i1 %cmp, label %for.end, label %for.body, !dbg !82

for.end:                                          ; preds = %for.body
  %sub = add nsw i32 %IntParI1, 4, !dbg !85
  %idxprom14 = sext i32 %sub to i64, !dbg !85
  %arrayidx17 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %idxprom14, !dbg !85
  %0 = load i32, i32* %arrayidx17, align 4, !dbg !85
  %inc18 = add nsw i32 %0, 1, !dbg !85
  store i32 %inc18, i32* %arrayidx17, align 4, !dbg !85
  %1 = load i32, i32* %arrayidx, align 4, !dbg !86
  %add22 = add nsw i32 %IntParI1, 25, !dbg !86
  %idxprom23 = sext i32 %add22 to i64, !dbg !86
  %arrayidx25 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom23, i64 %idxprom, !dbg !86
  store i32 %1, i32* %arrayidx25, align 4, !dbg !86
  store i32 5, i32* @IntGlob, align 4, !dbg !87
  ret void, !dbg !88
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind optsize }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!56}
!llvm.module.flags = !{!60}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "Version", scope: null, file: !2, line: 111, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "dry.c", directory: "/Users/manmanren/test-Nov/rdar_13183203/test2")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 32, align: 8, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!5 = !{!6}
!6 = !DISubrange(count: 4) ; [ DW_TAG_enumerator ] [Ident1 :: 0]
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression()) ; [ DW_TAG_enumerator ] [Ident2 :: 10000]
!8 = !DIGlobalVariable(name: "IntGlob", scope: null, file: !2, line: 171, type: !9, isLocal: false, isDefinition: true) ; [ DW_TAG_enumerator ] [Ident3 :: 10001]
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed) ; [ DW_TAG_enumerator ] [Ident4 :: 10002]
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression()) ; [ DW_TAG_enumerator ] [Ident5 :: 10003]
!11 = !DIGlobalVariable(name: "BoolGlob", scope: null, file: !2, line: 172, type: !12, isLocal: false, isDefinition: true)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !2, line: 149, baseType: !9)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = !DIGlobalVariable(name: "Char1Glob", scope: null, file: !2, line: 173, type: !4, isLocal: false, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = !DIGlobalVariable(name: "Char2Glob", scope: null, file: !2, line: 174, type: !4, isLocal: false, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = !DIGlobalVariable(name: "Array1Glob", scope: null, file: !2, line: 175, type: !19, isLocal: false, isDefinition: true)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "Array1Dim", file: !2, line: 135, baseType: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 1632, align: 32, elements: !21)
!21 = !{!22}
!22 = !DISubrange(count: 51)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = !DIGlobalVariable(name: "Array2Glob", scope: null, file: !2, line: 176, type: !25, isLocal: false, isDefinition: true)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "Array2Dim", file: !2, line: 136, baseType: !26)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 83232, align: 32, elements: !27)
!27 = !{!22, !22}
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = !DIGlobalVariable(name: "PtrGlb", scope: null, file: !2, line: 177, type: !30, isLocal: false, isDefinition: true)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "RecordPtr", file: !2, line: 148, baseType: !31)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 64, align: 64)
!32 = !DIDerivedType(tag: DW_TAG_typedef, name: "RecordType", file: !2, line: 147, baseType: !33)
!33 = !DICompositeType(tag: DW_TAG_structure_type, name: "Record", file: !2, line: 138, size: 448, align: 64, elements: !34)
!34 = !{!35, !37, !46, !47, !49}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "PtrComp", scope: !33, file: !2, line: 140, baseType: !36, size: 64, align: 64)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64, align: 64)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "Discr", scope: !33, file: !2, line: 141, baseType: !38, size: 32, align: 32, offset: 64)
!38 = !DIDerivedType(tag: DW_TAG_typedef, name: "Enumeration", file: !2, line: 128, baseType: !39)
!39 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !2, line: 128, size: 32, align: 32, elements: !40)
!40 = !{!41, !42, !43, !44, !45}
!41 = !DIEnumerator(name: "Ident1", value: 0)
!42 = !DIEnumerator(name: "Ident2", value: 10000)
!43 = !DIEnumerator(name: "Ident3", value: 10001)
!44 = !DIEnumerator(name: "Ident4", value: 10002)
!45 = !DIEnumerator(name: "Ident5", value: 10003)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "EnumComp", scope: !33, file: !2, line: 142, baseType: !38, size: 32, align: 32, offset: 96)
!47 = !DIDerivedType(tag: DW_TAG_member, name: "IntComp", scope: !33, file: !2, line: 143, baseType: !48, size: 32, align: 32, offset: 128)
!48 = !DIDerivedType(tag: DW_TAG_typedef, name: "OneToFifty", file: !2, line: 132, baseType: !9)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "StringComp", scope: !33, file: !2, line: 144, baseType: !50, size: 248, align: 8, offset: 160)
!50 = !DIDerivedType(tag: DW_TAG_typedef, name: "String30", file: !2, line: 134, baseType: !51)
!51 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 248, align: 8, elements: !52)
!52 = !{!53}
!53 = !DISubrange(count: 31)
!54 = !DIGlobalVariableExpression(var: !55, expr: !DIExpression())
!55 = !DIGlobalVariable(name: "PtrGlbNext", scope: null, file: !2, line: 178, type: !30, isLocal: false, isDefinition: true)
!56 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.3 (trunk 175015)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !57, retainedTypes: !58, globals: !59, imports: !58)
!57 = !{!39}
!58 = !{}
!59 = !{!0, !7, !10, !13, !15, !17, !23, !28, !54}
!60 = !{i32 1, !"Debug Info Version", i32 3}
!61 = distinct !DISubprogram(name: "Proc8", scope: !2, file: !2, line: 180, type: !62, isLocal: false, isDefinition: true, scopeLine: 185, virtualIndex: 6, isOptimized: true, unit: !56, retainedNodes: !66)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !64, !65, !48, !48}
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64, align: 64)
!66 = !{!67, !68, !69, !70, !71, !72}
!67 = !DILocalVariable(name: "Array1Par", arg: 1, scope: !61, file: !2, line: 181, type: !64)
!68 = !DILocalVariable(name: "Array2Par", arg: 2, scope: !61, file: !2, line: 182, type: !65)
!69 = !DILocalVariable(name: "IntParI1", arg: 3, scope: !61, file: !2, line: 183, type: !48)
!70 = !DILocalVariable(name: "IntParI2", arg: 4, scope: !61, file: !2, line: 184, type: !48)
!71 = !DILocalVariable(name: "IntLoc", scope: !61, file: !2, line: 186, type: !48)
!72 = !DILocalVariable(name: "IntIndex", scope: !61, file: !2, line: 187, type: !48)
!73 = !DIExpression()
!74 = !DILocation(line: 181, scope: !61)
!75 = !DILocation(line: 182, scope: !61)
!76 = !DILocation(line: 183, scope: !61)
!77 = !DILocation(line: 184, scope: !61)
!78 = !DILocation(line: 189, scope: !61)
!79 = !DILocation(line: 190, scope: !61)
!80 = !DILocation(line: 191, scope: !61)
!81 = !DILocation(line: 192, scope: !61)
!82 = !DILocation(line: 193, scope: !83)
!83 = distinct !DILexicalBlock(scope: !61, file: !2, line: 193)
!84 = !DILocation(line: 194, scope: !83)
!85 = !DILocation(line: 195, scope: !61)
!86 = !DILocation(line: 196, scope: !61)
!87 = !DILocation(line: 197, scope: !61)
!88 = !DILocation(line: 198, scope: !61)

