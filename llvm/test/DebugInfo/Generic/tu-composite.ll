; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
; CHECK: [[TYPE:.*]]: DW_TAG_structure_type
; Make sure we correctly handle containing type of a struct being a type identifier.
; CHECK-NEXT: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]} "C")
; CHECK-NEXT: DW_AT_name {{.*}}"C"

; Make sure we correctly handle context of a subprogram being a type identifier.
; CHECK: [[SP:.*]]: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}}"foo"
; Make sure we correctly handle containing type of a subprogram being a type identifier.
; CHECK: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]} "C")
; CHECK: DW_TAG_formal_parameter
; CHECK: NULL
; CHECK: NULL

; CHECK: [[TYPE2:.*]]: DW_TAG_structure_type
; CHECK: DW_AT_name {{.*}}"bar"
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name {{.*}}"D"
; CHECK: DW_TAG_member
; CHECK: DW_AT_name {{.*}}"a"
; Make sure we correctly handle context of a struct being a type identifier.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"Nested"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"Nested2"
; CHECK-NEXT: DW_AT_declaration [DW_FORM_flag]      (0x01)
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"virt<bar>"
; Make sure we correctly handle type of a template_type being a type identifier.
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]}
; CHECK-NEXT: DW_AT_name {{.*}}"T"
; Make sure we correctly handle derived-from of a typedef being a type identifier.
; CHECK: DW_TAG_typedef
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]}
; CHECK: DW_AT_name {{.*}}"baz2"
; Make sure we correctly handle derived-from of a pointer type being a type identifier.
; CHECK: DW_TAG_pointer_type
; CHECK: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE]]}
; CHECK: DW_TAG_typedef
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]}
; CHECK: DW_AT_name {{.*}}"baz"
; Make sure we correctly handle derived-from of an array type being a type identifier.
; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]}
; IR generated from clang -g with the following source:
; struct C {
;   virtual void foo();
; };
; void C::foo() {
; }
;
; struct bar { };
; typedef bar baz;
; struct D {
;   typedef bar baz2;
;   static int a;
;   struct Nested { };
;   struct Nested2 { };
;   template <typename T>
;   struct virt {
;     T* values;
;   };
; };
; void test() {
;   baz B;
;   bar A[3];
;   D::baz2 B2;
;   D::Nested e;
;   D::Nested2 *p;
;   D::virt<bar> t;
; }

%struct.C = type { ptr }
%struct.bar = type { i8 }
%"struct.D::Nested" = type { i8 }
%"struct.D::Nested2" = type { i8 }
%"struct.D::virt" = type { ptr }

@_ZTV1C = unnamed_addr constant [3 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1C3fooEv]
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS1C = constant [3 x i8] c"1C\00"
@_ZTI1C = unnamed_addr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1C }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1C3fooEv(ptr %this) unnamed_addr #0 align 2 !dbg !31 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !36, metadata !DIExpression()), !dbg !38
  %this1 = load ptr, ptr %this.addr
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_Z4testv() #0 !dbg !32 {
entry:
  %B = alloca %struct.bar, align 1
  %A = alloca [3 x %struct.bar], align 1
  %B2 = alloca %struct.bar, align 1
  %e = alloca %"struct.D::Nested", align 1
  %p = alloca ptr, align 8
  %t = alloca %"struct.D::virt", align 8
  call void @llvm.dbg.declare(metadata ptr %B, metadata !40, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.declare(metadata ptr %A, metadata !43, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata ptr %B2, metadata !48, metadata !DIExpression()), !dbg !50
  call void @llvm.dbg.declare(metadata ptr %e, metadata !51, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.declare(metadata ptr %p, metadata !53, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.declare(metadata ptr %t, metadata !56, metadata !DIExpression()), !dbg !57
  ret void, !dbg !58
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !59}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "tmp.cpp", directory: ".")
!2 = !{}
!3 = !{!4, !18, !19, !22, !23, !24}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", line: 1, file: !1, elements: !5, vtableHolder: !4, identifier: "_ZTS1C")
!5 = !{!6, !13}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$C", flags: DIFlagArtificial, file: !1, scope: !7, baseType: !8)
!7 = !DIFile(filename: "tmp.cpp", directory: ".")
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", size: 64, baseType: !10)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DISubprogram(name: "foo", linkageName: "_ZN1C3fooEv", line: 2, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !4, type: !14, containingType: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", line: 7, size: 8, align: 8, file: !1, elements: !2, identifier: "_ZTS3bar")
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "D", line: 9, size: 8, align: 8, file: !1, elements: !20, identifier: "_ZTS1D")
!20 = !{!21}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 11, flags: DIFlagStaticMember, file: !1, scope: !19, baseType: !12)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "Nested", line: 12, size: 8, align: 8, file: !1, scope: !19, elements: !2, identifier: "_ZTSN1D6NestedE")
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "Nested2", line: 13, flags: DIFlagFwdDecl, file: !1, scope: !19, identifier: "_ZTSN1D7Nested2E")
!24 = !DICompositeType(tag: DW_TAG_structure_type, name: "virt<bar>", line: 15, file: !1, scope: !19, elements: !25, templateParams: !28, identifier: "_ZTSN1D4virtI3barEE")
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "values", line: 16, size: 64, align: 64, file: !1, scope: !24, baseType: !27)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !18)
!28 = !{!29}
!29 = !DITemplateTypeParameter(name: "T", type: !18)
!31 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1C3fooEv", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: null, type: !14, declaration: !13, retainedNodes: !2)
!32 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", line: 20, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 20, file: !1, scope: !7, type: !33, retainedNodes: !2)
!33 = !DISubroutineType(types: !34)
!34 = !{null}
!35 = !{i32 2, !"Dwarf Version", i32 3}
!36 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !31, type: !37)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!38 = !DILocation(line: 0, scope: !31)
!39 = !DILocation(line: 5, scope: !31)
!40 = !DILocalVariable(name: "B", line: 21, scope: !32, file: !7, type: !41)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "baz", line: 8, file: !1, baseType: !18)
!42 = !DILocation(line: 21, scope: !32)
!43 = !DILocalVariable(name: "A", line: 22, scope: !32, file: !7, type: !44)
!44 = !DICompositeType(tag: DW_TAG_array_type, size: 24, align: 8, baseType: !18, elements: !45)
!45 = !{!46}
!46 = !DISubrange(count: 3)
!47 = !DILocation(line: 22, scope: !32)
!48 = !DILocalVariable(name: "B2", line: 23, scope: !32, file: !7, type: !49)
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "baz2", line: 10, file: !1, scope: !19, baseType: !18)
!50 = !DILocation(line: 23, scope: !32)
!51 = !DILocalVariable(name: "e", line: 24, scope: !32, file: !7, type: !22)
!52 = !DILocation(line: 24, scope: !32)
!53 = !DILocalVariable(name: "p", line: 25, scope: !32, file: !7, type: !54)
!54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23)
!55 = !DILocation(line: 25, scope: !32)
!56 = !DILocalVariable(name: "t", line: 26, scope: !32, file: !7, type: !24)
!57 = !DILocation(line: 26, scope: !32)
!58 = !DILocation(line: 27, scope: !32)
!59 = !{i32 1, !"Debug Info Version", i32 3}
