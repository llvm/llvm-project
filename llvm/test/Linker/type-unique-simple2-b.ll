; RUN: true
; This file belongs to type-unique-simple2-a.ll.
;
; $ cat b.cpp
; #include "ab.h"
; void A::setFoo() {}
; const
; foo_t A::getFoo() { return 1; }
; ModuleID = 'b.cpp'
; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
; target triple = "x86_64-apple-macosx10.9.0"

%class.A = type { i32 (...)** }

@_ZTV1A = unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1A6setFooEv to i8*), i8* bitcast (i32 (%class.A*)* @_ZN1A6getFooEv to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = constant [3 x i8] c"1A\00"
@_ZTI1A = unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }

; Function Attrs: nounwind
define void @_ZN1A6setFooEv(%class.A* %this) unnamed_addr #0 align 2 !dbg !26 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !32, metadata !DIExpression()), !dbg !34
  %this1 = load %class.A*, %class.A** %this.addr
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define i32 @_ZN1A6getFooEv(%class.A* %this) unnamed_addr #0 align 2 !dbg !28 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !36, metadata !DIExpression()), !dbg !37
  %this1 = load %class.A*, %class.A** %this.addr
  ret i32 1, !dbg !38
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, subprograms: !25, globals: !2, imports: !2)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 2, size: 64, align: 64, file: !5, elements: !6, vtableHolder: !"_ZTS1A", identifier: "_ZTS1A")
!5 = !DIFile(filename: "./ab.h", directory: "")
!6 = !{!7, !14, !19}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", size: 64, flags: DIFlagArtificial, file: !5, scope: !8, baseType: !9)
!8 = !DIFile(filename: "./ab.h", directory: "")
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", size: 64, baseType: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "setFoo", linkageName: "_ZN1A6setFooEv", line: 4, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !5, scope: !"_ZTS1A", type: !15, containingType: !"_ZTS1A")
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1A")
!19 = !DISubprogram(name: "getFoo", linkageName: "_ZN1A6getFooEv", line: 5, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !5, scope: !"_ZTS1A", type: !20, containingType: !"_ZTS1A")
!20 = !DISubroutineType(types: !21)
!21 = !{!22, !17}
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo_t", line: 1, file: !5, baseType: !13)
!25 = !{!26, !28}
!26 = distinct !DISubprogram(name: "setFoo", linkageName: "_ZN1A6setFooEv", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !27, scope: !"_ZTS1A", type: !15, declaration: !14, variables: !2)
!27 = !DIFile(filename: "b.cpp", directory: "")
!28 = distinct !DISubprogram(name: "getFoo", linkageName: "_ZN1A6getFooEv", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !27, scope: !"_ZTS1A", type: !20, declaration: !19, variables: !2)
!29 = !{i32 2, !"Dwarf Version", i32 2}
!30 = !{i32 1, !"Debug Info Version", i32 3}
!31 = !{!"clang version 3.5 "}
!32 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !26, type: !33)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1A")
!34 = !DILocation(line: 0, scope: !26)
!35 = !DILocation(line: 2, scope: !26)
!36 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !28, type: !33)
!37 = !DILocation(line: 0, scope: !28)
!38 = !DILocation(line: 4, scope: !28)
