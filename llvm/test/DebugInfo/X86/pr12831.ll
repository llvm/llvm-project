; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.function = type { i8 }
%class.BPLFunctionWriter = type { %struct.BPLModuleWriter* }
%struct.BPLModuleWriter = type { i8 }
%class.anon = type { i8 }
%class.anon.0 = type { i8 }

@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_" = internal alias void (%class.function*), void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"
@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_" = internal alias void (%class.function*), void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"

define void @_ZN17BPLFunctionWriter9writeExprEv(%class.BPLFunctionWriter* %this) nounwind uwtable align 2 !dbg !5 {
entry:
  %this.addr = alloca %class.BPLFunctionWriter*, align 8
  %agg.tmp = alloca %class.function, align 1
  %agg.tmp2 = alloca %class.anon, align 1
  %agg.tmp4 = alloca %class.function, align 1
  %agg.tmp5 = alloca %class.anon.0, align 1
  store %class.BPLFunctionWriter* %this, %class.BPLFunctionWriter** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.BPLFunctionWriter** %this.addr, metadata !133, metadata !DIExpression()), !dbg !135
  %this1 = load %class.BPLFunctionWriter*, %class.BPLFunctionWriter** %this.addr
  %MW = getelementptr inbounds %class.BPLFunctionWriter, %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !136
  %0 = load %struct.BPLModuleWriter*, %struct.BPLModuleWriter** %MW, align 8, !dbg !136
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %agg.tmp), !dbg !136
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %0), !dbg !136
  %MW3 = getelementptr inbounds %class.BPLFunctionWriter, %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !138
  %1 = load %struct.BPLModuleWriter*, %struct.BPLModuleWriter** %MW3, align 8, !dbg !138
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %agg.tmp4), !dbg !138
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %1), !dbg !138
  ret void, !dbg !139
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter*)

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon.0, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.function** %this.addr, metadata !140, metadata !DIExpression()), !dbg !142
  call void @llvm.dbg.declare(metadata %class.anon.0* %__f, metadata !143, metadata !DIExpression()), !dbg !144
  %this1 = load %class.function*, %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0* %__f), !dbg !145
  ret void, !dbg !147
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon.0*, align 8
  store %class.anon.0* %0, %class.anon.0** %.addr, align 8
  ret void, !dbg !148
}

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.function** %this.addr, metadata !150, metadata !DIExpression()), !dbg !151
  call void @llvm.dbg.declare(metadata %class.anon* %__f, metadata !152, metadata !DIExpression()), !dbg !153
  %this1 = load %class.function*, %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon* %__f), !dbg !154
  ret void, !dbg !156
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon*, align 8
  store %class.anon* %0, %class.anon** %.addr, align 8
  ret void, !dbg !157
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!162}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 ", isOptimized: false, emissionKind: FullDebug, file: !161, enums: !1, retainedTypes: !1, globals: !128)
!1 = !{}
!5 = distinct !DISubprogram(name: "writeExpr", linkageName: "_ZN17BPLFunctionWriter9writeExprEv", line: 19, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 19, file: !6, scope: null, type: !7, declaration: !103, variables: !1)
!6 = !DIFile(filename: "BPLFunctionWriter2.ii", directory: "/home/peter/crashdelta")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_class_type, name: "BPLFunctionWriter", line: 15, size: 64, align: 64, file: !160, elements: !11)
!11 = !{!12, !103}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "MW", line: 16, size: 64, align: 64, flags: DIFlagPrivate, file: !160, scope: !10, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_class_type, name: "BPLModuleWriter", line: 12, size: 8, align: 8, file: !160, elements: !15)
!15 = !{!16}
!16 = !DISubprogram(name: "writeIntrinsic", linkageName: "_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE", line: 13, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 13, file: !6, scope: !14, type: !17)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !20}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !14)
!20 = !DICompositeType(tag: DW_TAG_class_type, name: "function<void ()>", line: 6, size: 8, align: 8, file: !160, elements: !21, templateParams: !97)
!21 = !{!22, !51, !58, !86, !92}
!22 = !DISubprogram(name: "function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", line: 8, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !6, scope: !20, type: !23, templateParams: !47)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25, !26}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !20)
!26 = !DICompositeType(tag: DW_TAG_class_type, line: 20, size: 8, align: 8, file: !160, scope: !5, elements: !27)
!27 = !{!28, !35, !41}
!28 = !DISubprogram(name: "operator()", line: 20, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 20, file: !6, scope: !26, type: !29)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !32)
!32 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !26)
!35 = !DISubprogram(name: "~", line: 20, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 20, file: !6, scope: !26, type: !36)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !26)
!41 = !DISubprogram(name: "", line: 20, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 20, file: !6, scope: !26, type: !42)
!42 = !DISubroutineType(types: !43)
!43 = !{null, !38, !44}
!44 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !26)
!47 = !{!48}
!48 = !DITemplateTypeParameter(name: "_Functor", type: !26)
!51 = !DISubprogram(name: "function<function<void ()> >", line: 8, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !6, scope: !20, type: !52, templateParams: !54)
!52 = !DISubroutineType(types: !53)
!53 = !{null, !25, !20}
!54 = !{!55}
!55 = !DITemplateTypeParameter(name: "_Functor", type: !20)
!58 = !DISubprogram(name: "function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", line: 8, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 8, file: !6, scope: !20, type: !59, templateParams: !82)
!59 = !DISubroutineType(types: !60)
!60 = !{null, !25, !61}
!61 = !DICompositeType(tag: DW_TAG_class_type, line: 23, size: 8, align: 8, file: !160, scope: !5, elements: !62)
!62 = !{!63, !70, !76}
!63 = !DISubprogram(name: "operator()", line: 23, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 23, file: !6, scope: !61, type: !64)
!64 = !DISubroutineType(types: !65)
!65 = !{null, !66}
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !67)
!67 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !61)
!70 = !DISubprogram(name: "~", line: 23, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 23, file: !6, scope: !61, type: !71)
!71 = !DISubroutineType(types: !72)
!72 = !{null, !73}
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, baseType: !61)
!76 = !DISubprogram(name: "", line: 23, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 23, file: !6, scope: !61, type: !77)
!77 = !DISubroutineType(types: !78)
!78 = !{null, !73, !79}
!79 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !61)
!82 = !{!83}
!83 = !DITemplateTypeParameter(name: "_Functor", type: !61)
!86 = !DISubprogram(name: "function", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !6, scope: !20, type: !87)
!87 = !DISubroutineType(types: !88)
!88 = !{null, !25, !89}
!89 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !20)
!92 = !DISubprogram(name: "~function", line: 6, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 6, file: !6, scope: !20, type: !93)
!93 = !DISubroutineType(types: !94)
!94 = !{null, !25}
!97 = !{!98}
!98 = !DITemplateTypeParameter(name: "T", type: !99)
!99 = !DISubroutineType(types: !100)
!100 = !{null}
!103 = !DISubprogram(name: "writeExpr", linkageName: "_ZN17BPLFunctionWriter9writeExprEv", line: 17, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate | DIFlagPrototyped, isOptimized: false, scopeLine: 17, file: !6, scope: !10, type: !7)
!106 = distinct !DISubprogram(name: "function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", linkageName: "_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_", line: 8, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 8, file: !6, scope: null, type: !59, templateParams: !82, declaration: !58, variables: !1)
!107 = distinct !DISubprogram(name: "_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", linkageName: "_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !6, scope: null, type: !108, templateParams: !111, declaration: !113, variables: !1)
!108 = !DISubroutineType(types: !109)
!109 = !{null, !110}
!110 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !61)
!111 = !{!112}
!112 = !DITemplateTypeParameter(name: "_Tp", type: !61)
!113 = !DISubprogram(name: "_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", linkageName: "_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !6, scope: !114, type: !108, templateParams: !111)
!114 = !DICompositeType(tag: DW_TAG_class_type, name: "_Base_manager", line: 1, size: 8, align: 8, file: !160, elements: !115)
!115 = !{!116, !113}
!116 = !DISubprogram(name: "_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", linkageName: "_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !6, scope: !114, type: !117, templateParams: !120)
!117 = !DISubroutineType(types: !118)
!118 = !{null, !119}
!119 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !26)
!120 = !{!121}
!121 = !DITemplateTypeParameter(name: "_Tp", type: !26)
!126 = distinct !DISubprogram(name: "function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", linkageName: "_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_", line: 8, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 8, file: !6, scope: null, type: !23, templateParams: !47, declaration: !22, variables: !1)
!127 = distinct !DISubprogram(name: "_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", linkageName: "_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !6, scope: null, type: !117, templateParams: !120, declaration: !116, variables: !1)
!128 = !{!130}
!130 = !DIGlobalVariable(name: "__stored_locally", linkageName: "__stored_locally", line: 2, isLocal: true, isDefinition: true, scope: !114, file: !6, type: !131, variable: i1 1)
!131 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !132)
!132 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!133 = !DILocalVariable(name: "this", line: 19, arg: 1, flags: DIFlagArtificial, scope: !5, file: !6, type: !134)
!134 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !10)
!135 = !DILocation(line: 19, column: 39, scope: !5)
!136 = !DILocation(line: 20, column: 17, scope: !137)
!137 = distinct !DILexicalBlock(line: 19, column: 51, file: !6, scope: !5)
!138 = !DILocation(line: 23, column: 17, scope: !137)
!139 = !DILocation(line: 26, column: 15, scope: !137)
!140 = !DILocalVariable(name: "this", line: 8, arg: 1, flags: DIFlagArtificial, scope: !106, file: !6, type: !141)
!141 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !20)
!142 = !DILocation(line: 8, column: 45, scope: !106)
!143 = !DILocalVariable(name: "__f", line: 8, arg: 2, scope: !106, file: !6, type: !61)
!144 = !DILocation(line: 8, column: 63, scope: !106)
!145 = !DILocation(line: 9, column: 9, scope: !146)
!146 = distinct !DILexicalBlock(line: 8, column: 81, file: !6, scope: !106)
!147 = !DILocation(line: 10, column: 13, scope: !146)
!148 = !DILocation(line: 4, column: 5, scope: !149)
!149 = distinct !DILexicalBlock(line: 3, column: 105, file: !6, scope: !107)
!150 = !DILocalVariable(name: "this", line: 8, arg: 1, flags: DIFlagArtificial, scope: !126, file: !6, type: !141)
!151 = !DILocation(line: 8, column: 45, scope: !126)
!152 = !DILocalVariable(name: "__f", line: 8, arg: 2, scope: !126, file: !6, type: !26)
!153 = !DILocation(line: 8, column: 63, scope: !126)
!154 = !DILocation(line: 9, column: 9, scope: !155)
!155 = distinct !DILexicalBlock(line: 8, column: 81, file: !6, scope: !126)
!156 = !DILocation(line: 10, column: 13, scope: !155)
!157 = !DILocation(line: 4, column: 5, scope: !158)
!158 = distinct !DILexicalBlock(line: 3, column: 105, file: !6, scope: !127)
!159 = !DIFile(filename: "BPLFunctionWriter.cpp", directory: "/home/peter/crashdelta")
!160 = !DIFile(filename: "BPLFunctionWriter2.ii", directory: "/home/peter/crashdelta")
!161 = !DIFile(filename: "BPLFunctionWriter.cpp", directory: "/home/peter/crashdelta")
!162 = !{i32 1, !"Debug Info Version", i32 3}
