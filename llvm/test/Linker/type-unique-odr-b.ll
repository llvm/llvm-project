; RUN: true
; This file belongs to type-unique-odr-a.ll.
;
; Test ODR-based type uniquing for C++ class members.
; rdar://problem/15851313.
;
; $ cat -n type-unique-odr-b.cpp
;     1	// Make this declaration start on a different line.
;     2	class A {
;     3	  int data;
;     4	protected:
;     5	  void getFoo();
;     6	};
;     7
;     8	void A::getFoo() {}
;     9
;    10	static void bar() {}
;    11	void f() { bar(); };

; ModuleID = 'type-unique-odr-b.cpp'

%class.A = type { i32 }

; Function Attrs: nounwind
define void @_ZN1A6getFooEv(%class.A* %this) #0 align 2 !dbg !15 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !24, metadata !DIExpression()), !dbg !26
  %this1 = load %class.A*, %class.A** %this.addr
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define void @_Z1fv() #0 !dbg !16 {
entry:
  call void @_ZL3barv(), !dbg !28
  ret void, !dbg !28
}

; Function Attrs: nounwind
define internal void @_ZL3barv() #0 !dbg !20 {
entry:
  ret void, !dbg !29
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 2, size: 32, align: 32, file: !5, elements: !6, identifier: "_ZTS1A")
!5 = !DIFile(filename: "type-unique-odr-b.cpp", directory: "")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "data", line: 3, size: 32, align: 32, flags: DIFlagPrivate, file: !5, scope: !4, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DISubprogram(name: "getFoo", linkageName: "_ZN1A6getFooEv", line: 5, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: false, scopeLine: 5, file: !5, scope: !4, type: !10)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!15 = distinct !DISubprogram(name: "getFoo", linkageName: "_ZN1A6getFooEv", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 8, file: !5, scope: !4, type: !10, declaration: !9, variables: !2)
!16 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 11, file: !5, scope: !17, type: !18, variables: !2)
!17 = !DIFile(filename: "type-unique-odr-b.cpp", directory: "")
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = distinct !DISubprogram(name: "bar", linkageName: "_ZL3barv", line: 10, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 10, file: !5, scope: !17, type: !18, variables: !2)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5.0 "}
!24 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !15, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!26 = !DILocation(line: 0, scope: !15)
!27 = !DILocation(line: 8, scope: !15)
!28 = !DILocation(line: 11, scope: !16)
!29 = !DILocation(line: 10, scope: !20)
