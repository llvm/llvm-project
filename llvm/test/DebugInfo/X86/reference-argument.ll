; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=asm %s -o - | FileCheck %s
; ModuleID = 'aggregate-indirect-arg.cpp'
; extracted from debuginfo-tests/aggregate-indirect-arg.cpp

; v should not be a pointer.
; CHECK: ##DEBUG_VALUE: foo:v <- %RSI
; rdar://problem/13658587

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%class.SVal = type { i8*, i32 }
%class.A = type { i8 }

declare void @_Z3barR4SVal(%class.SVal* %v)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare i32 @main()
; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1A3fooE4SVal(%class.A* %this, %class.SVal* %v) nounwind ssp uwtable align 2 !dbg !35 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !59, metadata !DIExpression()), !dbg !61
  call void @llvm.dbg.declare(metadata %class.SVal* %v, metadata !62, metadata !DIExpression(DW_OP_deref)), !dbg !61
  %this1 = load %class.A*, %class.A** %this.addr
  call void @_Z3barR4SVal(%class.SVal* %v), !dbg !61
  ret void, !dbg !61
}
declare void @_ZN4SValD1Ev(%class.SVal* %this)
declare void @_ZN4SValD2Ev(%class.SVal* %this)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !68}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "aggregate-indirect-arg.cpp", directory: "")
!2 = !{}
!3 = !{!4, !29, !33, !34, !35}
!4 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barR4SVal", line: 19, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 19, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "aggregate-indirect-arg.cpp", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_class_type, name: "SVal", line: 12, size: 128, align: 64, file: !1, elements: !10)
!10 = !{!11, !14, !16, !21, !23}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "Data", line: 15, size: 64, align: 64, file: !1, scope: !9, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "Kind", line: 16, size: 32, align: 32, offset: 64, file: !1, scope: !9, baseType: !15)
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!16 = !DISubprogram(name: "~SVal", line: 14, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 14, file: !1, scope: !9, type: !17)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !9)
!21 = !DISubprogram(name: "SVal", line: 12, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 12, file: !1, scope: !9, type: !17)
!23 = !DISubprogram(name: "SVal", line: 12, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 12, file: !1, scope: !9, type: !24)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !19, !26}
!26 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !27)
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!29 = distinct !DISubprogram(name: "main", line: 25, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 25, file: !1, scope: !5, type: !30, variables: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{!32}
!32 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!33 = distinct !DISubprogram(name: "~SVal", linkageName: "_ZN4SValD1Ev", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 14, file: !1, scope: null, type: !17, declaration: !16, variables: !2)
!34 = distinct !DISubprogram(name: "~SVal", linkageName: "_ZN4SValD2Ev", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 14, file: !1, scope: null, type: !17, declaration: !16, variables: !2)
!35 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1A3fooE4SVal", line: 22, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 22, file: !1, scope: null, type: !36, declaration: !41, variables: !2)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !38, !9}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !39)
!39 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 20, size: 8, align: 8, file: !1, elements: !40)
!40 = !{!41, !43}
!41 = !DISubprogram(name: "foo", linkageName: "_ZN1A3fooE4SVal", line: 22, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 22, file: !1, scope: !39, type: !36)
!43 = !DISubprogram(name: "A", line: 20, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 20, file: !1, scope: !39, type: !44)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !38}
!47 = !{i32 2, !"Dwarf Version", i32 3}
!48 = !DILocalVariable(name: "v", line: 19, arg: 1, scope: !4, file: !5, type: !8)
!49 = !DILocation(line: 19, scope: !4)
!50 = !DILocalVariable(name: "v", line: 26, scope: !29, file: !5, type: !9)
!51 = !DILocation(line: 26, scope: !29)
!52 = !DILocation(line: 27, scope: !29)
!53 = !DILocation(line: 28, scope: !29)
!54 = !DILocalVariable(name: "a", line: 29, scope: !29, file: !5, type: !39)
!55 = !DILocation(line: 29, scope: !29)
!56 = !DILocation(line: 30, scope: !29)
!57 = !DILocation(line: 31, scope: !29)
!58 = !DILocation(line: 32, scope: !29)
!59 = !DILocalVariable(name: "this", line: 22, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !35, file: !5, type: !60)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !39)
!61 = !DILocation(line: 22, scope: !35)
!62 = !DILocalVariable(name: "v", line: 22, arg: 2, scope: !35, file: !5, type: !9)
!63 = !DILocalVariable(name: "this", line: 14, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !33, file: !5, type: !64)
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!65 = !DILocation(line: 14, scope: !33)
!66 = !DILocalVariable(name: "this", line: 14, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !34, file: !5, type: !64)
!67 = !DILocation(line: 14, scope: !34)
!68 = !{i32 1, !"Debug Info Version", i32 3}
