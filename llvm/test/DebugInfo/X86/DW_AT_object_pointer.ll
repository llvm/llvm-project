; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_formal_parameter [
; CHECK-NOT: ""
; CHECK: DW_TAG
; CHECK: DW_TAG_class_type
; CHECK: DW_AT_object_pointer [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]*}} => {[[PARAM:0x[0-9a-f]*]]})
; CHECK: [[PARAM]]:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x{{[0-9a-f]*}}] = "this")

%class.A = type { i32 }

define i32 @_Z3fooi(i32) nounwind uwtable ssp !dbg !5 {
entry:
  %.addr = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !36, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !21, metadata !DIExpression()), !dbg !23
  call void @_ZN1AC1Ev(%class.A* %a), !dbg !24
  %m_a = getelementptr inbounds %class.A, %class.A* %a, i32 0, i32 0, !dbg !25
  %1 = load i32, i32* %m_a, align 4, !dbg !25
  ret i32 %1, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 !dbg !10 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !26, metadata !DIExpression()), !dbg !28
  %this1 = load %class.A*, %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1), !dbg !29
  ret void, !dbg !29
}

define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 !dbg !20 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !30, metadata !DIExpression()), !dbg !31
  %this1 = load %class.A*, %class.A** %this.addr
  %m_a = getelementptr inbounds %class.A, %class.A* %this1, i32 0, i32 0, !dbg !32
  store i32 0, i32* %m_a, align 4, !dbg !32
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 163586) (llvm/trunk 163570)", isOptimized: false, emissionKind: FullDebug, file: !37, enums: !1, retainedTypes: !1, subprograms: !3, globals: !1, imports:  !1)
!1 = !{}
!3 = !{!5, !10, !20}
!5 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !6, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "bar.cpp", directory: "/Users/echristo/debug-tests")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC1Ev", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !6, scope: null, type: !11, declaration: !17, variables: !1)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 1, size: 32, align: 32, file: !37, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "m_a", line: 4, size: 32, align: 32, file: !37, scope: !14, baseType: !9)
!17 = !DISubprogram(name: "A", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !6, scope: !14, type: !11)
!20 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC2Ev", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !6, scope: null, type: !11, declaration: !17, variables: !1)
!21 = !DILocalVariable(name: "a", line: 8, scope: !22, file: !6, type: !14)
!22 = distinct !DILexicalBlock(line: 7, column: 11, file: !6, scope: !5)
!23 = !DILocation(line: 8, column: 5, scope: !22)
!24 = !DILocation(line: 8, column: 6, scope: !22)
!25 = !DILocation(line: 9, column: 3, scope: !22)
!26 = !DILocalVariable(name: "this", line: 3, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !10, file: !6, type: !27)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !14)
!28 = !DILocation(line: 3, column: 3, scope: !10)
!29 = !DILocation(line: 3, column: 18, scope: !10)
!30 = !DILocalVariable(name: "this", line: 3, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !20, file: !6, type: !27)
!31 = !DILocation(line: 3, column: 3, scope: !20)
!32 = !DILocation(line: 3, column: 9, scope: !33)
!33 = distinct !DILexicalBlock(line: 3, column: 7, file: !6, scope: !20)
!34 = !DILocation(line: 3, column: 18, scope: !33)
!35 = !DILocation(line: 7, scope: !5)
!36 = !DILocalVariable(name: "", line: 7, arg: 1, scope: !5, file: !6, type: !9)
!37 = !DIFile(filename: "bar.cpp", directory: "/Users/echristo/debug-tests")
!38 = !{i32 1, !"Debug Info Version", i32 3}
