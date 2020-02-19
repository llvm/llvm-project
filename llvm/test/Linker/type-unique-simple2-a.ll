; REQUIRES: default_triple
;
; RUN: llvm-link %s %p/type-unique-simple2-b.ll -S -o - | %llc_dwarf -filetype=obj -O0 | llvm-dwarfdump -v -debug-info - | FileCheck %s
;
; Tests for a merge error where attributes are inserted twice into the same DIE.
;
; $ cat ab.h
; typedef int foo_t;
; class A {
; public:
;   virtual void setFoo();
;   virtual const foo_t getFoo();
; };
;
; $ cat a.cpp
; #include "ab.h"
; foo_t bar() {
;     return A().getFoo();
; }
;
; CHECK: DW_AT_name {{.*}} "setFoo"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG

; ModuleID = 'a.cpp'

%class.A = type { i32 (...)** }

@_ZTV1A = external unnamed_addr constant [4 x i8*]

; Function Attrs: nounwind
define i32 @_Z3barv() #0 !dbg !27 {
entry:
  %tmp = alloca %class.A, align 8
  %0 = bitcast %class.A* %tmp to i8*, !dbg !38
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 8, i1 false), !dbg !38
  call void @_ZN1AC1Ev(%class.A* %tmp) #1, !dbg !38
  %call = call i32 @_ZN1A6getFooEv(%class.A* %tmp), !dbg !38
  ret i32 %call, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #1

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr #2 align 2 !dbg !31 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !39, metadata !DIExpression()), !dbg !41
  %this1 = load %class.A*, %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1) #1, !dbg !42
  ret void, !dbg !42
}

declare i32 @_ZN1A6getFooEv(%class.A*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #4

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr #2 align 2 !dbg !34 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !44, metadata !DIExpression()), !dbg !45
  %this1 = load %class.A*, %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !46
  store i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !46
  ret void, !dbg !46
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { inlinehint nounwind }
attributes #4 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36}
!llvm.ident = !{!37}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 2, size: 64, align: 64, file: !5, elements: !6, vtableHolder: !4, identifier: "_ZTS1A")
!5 = !DIFile(filename: "./ab.h", directory: "")
!6 = !{!7, !14, !19}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", size: 64, flags: DIFlagArtificial, file: !5, scope: !8, baseType: !9)
!8 = !DIFile(filename: "./ab.h", directory: "")
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", size: 64, baseType: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "setFoo", linkageName: "_ZN1A6setFooEv", line: 4, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !5, scope: !4, type: !15, containingType: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!19 = !DISubprogram(name: "getFoo", linkageName: "_ZN1A6getFooEv", line: 5, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !5, scope: !4, type: !20, containingType: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{!22, !17}
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo_t", line: 1, file: !24, baseType: !13)
!24 = !DIFile(filename: "a.cpp", directory: "")
!27 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !24, scope: !28, type: !29, retainedNodes: !2)
!28 = !DIFile(filename: "a.cpp", directory: "")
!29 = !DISubroutineType(types: !30)
!30 = !{!23}
!31 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC1Ev", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !5, scope: !4, type: !15, declaration: !32, retainedNodes: !2)
!32 = !DISubprogram(name: "A", isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scope: !4, type: !15)
!34 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC2Ev", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !5, scope: !4, type: !15, declaration: !32, retainedNodes: !2)
!35 = !{i32 2, !"Dwarf Version", i32 2}
!36 = !{i32 1, !"Debug Info Version", i32 3}
!37 = !{!"clang version 3.5 "}
!38 = !DILocation(line: 3, scope: !27)
!39 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !31, type: !40)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!41 = !DILocation(line: 0, scope: !31)
!42 = !DILocation(line: 2, scope: !43)
!43 = !DILexicalBlockFile(discriminator: 0, file: !5, scope: !31)
!44 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !34, type: !40)
!45 = !DILocation(line: 0, scope: !34)
!46 = !DILocation(line: 2, scope: !34)
