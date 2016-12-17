; RUN: llc %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -enable-misched
; RUN: llvm-dwarfdump %t | FileCheck %s

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

%struct.Record = type { %struct.Record*, i32, i32, i32, [31 x i8] }

@Version = global [4 x i8] c"1.1\00", align 1, !dbg !30
@IntGlob = common global i32 0, align 4, !dbg !35
@BoolGlob = common global i32 0, align 4, !dbg !36
@Char1Glob = common global i8 0, align 1, !dbg !38
@Char2Glob = common global i8 0, align 1, !dbg !39
@Array1Glob = common global [51 x i32] zeroinitializer, align 16, !dbg !40
@Array2Glob = common global [51 x [51 x i32]] zeroinitializer, align 16, !dbg !42
@PtrGlb = common global %struct.Record* null, align 8, !dbg !46
@PtrGlbNext = common global %struct.Record* null, align 8, !dbg !63

define void @Proc8(i32* nocapture %Array1Par, [51 x i32]* nocapture %Array2Par, i32 %IntParI1, i32 %IntParI2) nounwind optsize !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %Array1Par, i64 0, metadata !23, metadata !DIExpression()), !dbg !64
  tail call void @llvm.dbg.value(metadata [51 x i32]* %Array2Par, i64 0, metadata !24, metadata !DIExpression()), !dbg !65
  tail call void @llvm.dbg.value(metadata i32 %IntParI1, i64 0, metadata !25, metadata !DIExpression()), !dbg !66
  tail call void @llvm.dbg.value(metadata i32 %IntParI2, i64 0, metadata !26, metadata !DIExpression()), !dbg !67
  %add = add i32 %IntParI1, 5, !dbg !68
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !27, metadata !DIExpression()), !dbg !68
  %idxprom = sext i32 %add to i64, !dbg !69
  %arrayidx = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom, !dbg !69
  store i32 %IntParI2, i32* %arrayidx, align 4, !dbg !69
  %add3 = add nsw i32 %IntParI1, 6, !dbg !73
  %idxprom4 = sext i32 %add3 to i64, !dbg !73
  %arrayidx5 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom4, !dbg !73
  store i32 %IntParI2, i32* %arrayidx5, align 4, !dbg !73
  %add6 = add nsw i32 %IntParI1, 35, !dbg !74
  %idxprom7 = sext i32 %add6 to i64, !dbg !74
  %arrayidx8 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom7, !dbg !74
  store i32 %add, i32* %arrayidx8, align 4, !dbg !74
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !28, metadata !DIExpression()), !dbg !75
  br label %for.body, !dbg !75

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %idxprom, %entry ], [ %indvars.iv.next, %for.body ]
  %IntIndex.046 = phi i32 [ %add, %entry ], [ %inc, %for.body ]
  %arrayidx13 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %indvars.iv, !dbg !77
  store i32 %add, i32* %arrayidx13, align 4, !dbg !77
  %inc = add nsw i32 %IntIndex.046, 1, !dbg !75
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !28, metadata !DIExpression()), !dbg !75
  %cmp = icmp sgt i32 %inc, %add3, !dbg !75
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !75
  br i1 %cmp, label %for.end, label %for.body, !dbg !75

for.end:                                          ; preds = %for.body
  %sub = add nsw i32 %IntParI1, 4, !dbg !78
  %idxprom14 = sext i32 %sub to i64, !dbg !78
  %arrayidx17 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %idxprom14, !dbg !78
  %0 = load i32, i32* %arrayidx17, align 4, !dbg !78
  %inc18 = add nsw i32 %0, 1, !dbg !78
  store i32 %inc18, i32* %arrayidx17, align 4, !dbg !78
  %1 = load i32, i32* %arrayidx, align 4, !dbg !79
  %add22 = add nsw i32 %IntParI1, 25, !dbg !79
  %idxprom23 = sext i32 %add22 to i64, !dbg !79
  %arrayidx25 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom23, i64 %idxprom, !dbg !79
  store i32 %1, i32* %arrayidx25, align 4, !dbg !79
  store i32 5, i32* @IntGlob, align 4, !dbg !80
  ret void, !dbg !81
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

attributes #0 = { nounwind optsize ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 175015)", isOptimized: true, emissionKind: FullDebug, file: !82, enums: !1, retainedTypes: !10, globals: !29, imports:  !10)
!1 = !{!2}
!2 = !DICompositeType(tag: DW_TAG_enumeration_type, line: 128, size: 32, align: 32, file: !82, elements: !4)
!3 = !DIFile(filename: "dry.c", directory: "/Users/manmanren/test-Nov/rdar_13183203/test2")
!4 = !{!5, !6, !7, !8, !9}
!5 = !DIEnumerator(name: "Ident1", value: 0) ; [ DW_TAG_enumerator ] [Ident1 :: 0]
!6 = !DIEnumerator(name: "Ident2", value: 10000) ; [ DW_TAG_enumerator ] [Ident2 :: 10000]
!7 = !DIEnumerator(name: "Ident3", value: 10001) ; [ DW_TAG_enumerator ] [Ident3 :: 10001]
!8 = !DIEnumerator(name: "Ident4", value: 10002) ; [ DW_TAG_enumerator ] [Ident4 :: 10002]
!9 = !DIEnumerator(name: "Ident5", value: 10003) ; [ DW_TAG_enumerator ] [Ident5 :: 10003]
!10 = !{}
!12 = distinct !DISubprogram(name: "Proc8", line: 180, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 185, file: !82, scope: !3, type: !13, variables: !22)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15, !17, !21, !21}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !16)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, size: 1632, align: 32, baseType: !16, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 51)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "OneToFifty", line: 132, file: !82, baseType: !16)
!22 = !{!23, !24, !25, !26, !27, !28}
!23 = !DILocalVariable(name: "Array1Par", line: 181, arg: 1, scope: !12, file: !3, type: !15)
!24 = !DILocalVariable(name: "Array2Par", line: 182, arg: 2, scope: !12, file: !3, type: !17)
!25 = !DILocalVariable(name: "IntParI1", line: 183, arg: 3, scope: !12, file: !3, type: !21)
!26 = !DILocalVariable(name: "IntParI2", line: 184, arg: 4, scope: !12, file: !3, type: !21)
!27 = !DILocalVariable(name: "IntLoc", line: 186, scope: !12, file: !3, type: !21)
!28 = !DILocalVariable(name: "IntIndex", line: 187, scope: !12, file: !3, type: !21)
!29 = !{!30, !35, !36, !38, !39, !40, !42, !46, !63}
!30 = !DIGlobalVariable(name: "Version", line: 111, isLocal: false, isDefinition: true, scope: null, file: !3, type: !31)
!31 = !DICompositeType(tag: DW_TAG_array_type, size: 32, align: 8, baseType: !32, elements: !33)
!32 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!33 = !{!34}
!34 = !DISubrange(count: 4)
!35 = !DIGlobalVariable(name: "IntGlob", line: 171, isLocal: false, isDefinition: true, scope: null, file: !3, type: !16)
!36 = !DIGlobalVariable(name: "BoolGlob", line: 172, isLocal: false, isDefinition: true, scope: null, file: !3, type: !37)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", line: 149, file: !82, baseType: !16)
!38 = !DIGlobalVariable(name: "Char1Glob", line: 173, isLocal: false, isDefinition: true, scope: null, file: !3, type: !32)
!39 = !DIGlobalVariable(name: "Char2Glob", line: 174, isLocal: false, isDefinition: true, scope: null, file: !3, type: !32)
!40 = !DIGlobalVariable(name: "Array1Glob", line: 175, isLocal: false, isDefinition: true, scope: null, file: !3, type: !41)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "Array1Dim", line: 135, file: !82, baseType: !18)
!42 = !DIGlobalVariable(name: "Array2Glob", line: 176, isLocal: false, isDefinition: true, scope: null, file: !3, type: !43)
!43 = !DIDerivedType(tag: DW_TAG_typedef, name: "Array2Dim", line: 136, file: !82, baseType: !44)
!44 = !DICompositeType(tag: DW_TAG_array_type, size: 83232, align: 32, baseType: !16, elements: !45)
!45 = !{!20, !20}
!46 = !DIGlobalVariable(name: "PtrGlb", line: 177, isLocal: false, isDefinition: true, scope: null, file: !3, type: !47)
!47 = !DIDerivedType(tag: DW_TAG_typedef, name: "RecordPtr", line: 148, file: !82, baseType: !48)
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !49)
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "RecordType", line: 147, file: !82, baseType: !50)
!50 = !DICompositeType(tag: DW_TAG_structure_type, name: "Record", line: 138, size: 448, align: 64, file: !82, elements: !51)
!51 = !{!52, !54, !56, !57, !58}
!52 = !DIDerivedType(tag: DW_TAG_member, name: "PtrComp", line: 140, size: 64, align: 64, file: !82, scope: !50, baseType: !53)
!53 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !50)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "Discr", line: 141, size: 32, align: 32, offset: 64, file: !82, scope: !50, baseType: !55)
!55 = !DIDerivedType(tag: DW_TAG_typedef, name: "Enumeration", line: 128, file: !82, baseType: !2)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "EnumComp", line: 142, size: 32, align: 32, offset: 96, file: !82, scope: !50, baseType: !55)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "IntComp", line: 143, size: 32, align: 32, offset: 128, file: !82, scope: !50, baseType: !21)
!58 = !DIDerivedType(tag: DW_TAG_member, name: "StringComp", line: 144, size: 248, align: 8, offset: 160, file: !82, scope: !50, baseType: !59)
!59 = !DIDerivedType(tag: DW_TAG_typedef, name: "String30", line: 134, file: !82, baseType: !60)
!60 = !DICompositeType(tag: DW_TAG_array_type, size: 248, align: 8, baseType: !32, elements: !61)
!61 = !{!62}
!62 = !DISubrange(count: 31)
!63 = !DIGlobalVariable(name: "PtrGlbNext", line: 178, isLocal: false, isDefinition: true, scope: null, file: !3, type: !47)
!64 = !DILocation(line: 181, scope: !12)
!65 = !DILocation(line: 182, scope: !12)
!66 = !DILocation(line: 183, scope: !12)
!67 = !DILocation(line: 184, scope: !12)
!68 = !DILocation(line: 189, scope: !12)
!69 = !DILocation(line: 190, scope: !12)
!73 = !DILocation(line: 191, scope: !12)
!74 = !DILocation(line: 192, scope: !12)
!75 = !DILocation(line: 193, scope: !76)
!76 = distinct !DILexicalBlock(line: 193, column: 0, file: !82, scope: !12)
!77 = !DILocation(line: 194, scope: !76)
!78 = !DILocation(line: 195, scope: !12)
!79 = !DILocation(line: 196, scope: !12)
!80 = !DILocation(line: 197, scope: !12)
!81 = !DILocation(line: 198, scope: !12)
!82 = !DIFile(filename: "dry.c", directory: "/Users/manmanren/test-Nov/rdar_13183203/test2")
!83 = !{i32 1, !"Debug Info Version", i32 3}
