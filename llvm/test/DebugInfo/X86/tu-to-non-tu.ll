; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; This test was originally for testing references to non-TU types from TUs, but
; it's currently too complicated to get that right reliably - non-TU types may
; have local linkage so declaring them in a TU is unreliable (consumers would
; have to be context-dependent to know which CU the TU is referenced from to
; know which type definition to match the declaration in the TU up to)

; Might as well keep the complex cases here in case we revisit this (with a flag
; on the type to indicate whether it should be in a type unit or not, distinct
; from the mangled name indicating whether the type has linkage or not perhaps
; (rather than checing the namespace scopes to see if any are internal (which
; doesn't handle the linkage of, say, a template instantiated with an internal
; type))

; Built from the following source, compiled with this command:
; $ clang++-tot decl.cpp -g -fdebug-types-section -c
; And modified (as noted in the comments) to produce some "simplified" and "mangled"
; simplified template names, to ensure they get template parameters in declarations
; created in type units.

; struct non_tu {
;   virtual void f1();
; };
; void non_tu::f1() {}
; struct tu_ref_non_tu {
;   non_tu v1;
; };
; tu_ref_non_tu v1;
; 
; // Reference internal 
; namespace {
; struct internal {};
; }  // namespace
; struct ref_internal {
;   internal i;
; };
; ref_internal v5;
; 
; 
; template <typename T>
; struct templ_non_tu;
;
; // Reference to (normal, non-mangled/simplified) non-tu type with template
; // parameters.
; template <>
; struct templ_non_tu<int> {
;   virtual void f1();
; };
; void templ_non_tu<int>::f1() {}
; struct ref_templ_non_tu {
;   templ_non_tu<int> v1;
; };
; ref_templ_non_tu v2;
; 
; // Modify templ_non_tu<long>'s name to be simplified (strip template parameter
; // list from the "name" attribute)
; template <>
; struct templ_non_tu<long> {
;   virtual void f1();
; };
; void templ_non_tu<long>::f1() {}
; struct ref_templ_non_tu_simple {
;   templ_non_tu<long> v1;
; };
; ref_templ_non_tu_simple v3;
; 
; // Modify templ_non_tu<bool>'s name to be mangled ('_STN' name '|' args)
; template <>
; struct templ_non_tu<bool> {
;   virtual void f1();
; };
; void templ_non_tu<bool>::f1() {}
; struct ref_templ_non_tu_mangled {
;   templ_non_tu<bool> v1;
; };
; ref_templ_non_tu_mangled v4;




; CHECK-LABEL: Type Unit:
; CHECK: DW_TAG_class_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_from_ref_internal_template"


; CHECK-LABEL: Compile Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"tu_ref_non_tu"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"non_tu"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_internal"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_byte_size

; CHECK: DW_TAG_namespace
; CHECK-NOT: {{DW_TAG|DW_AT}}
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"internal"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"internal_template<ref_from_ref_internal_template>"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu"
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"_STNtempl_non_tu|<int>"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu_simple"
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name      ("templ_non_tu")
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type    {{.*}}"long"

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu_mangled"
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name      ("_STNtempl_non_tu|<bool>")
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type    {{.*}}"bool"
; CHECK: DW_TAG_class_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_internal_template"
; CHECK: DW_TAG_class_type
; CHECK-NEXT:   DW_AT_declaration       (true)
; CHECK-NEXT:   DW_AT_signature (0x30c4e2370930c7ad)

%struct.ref_internal = type { %"struct.(anonymous namespace)::internal" }
%"struct.(anonymous namespace)::internal" = type { i8 }
%class.ref_internal_template = type { %"struct.(anonymous namespace)::internal_template" }
%"struct.(anonymous namespace)::internal_template" = type { i8 }
%class.ref_from_ref_internal_template = type { i8 }
%struct.non_tu = type { i32 (...)** }
%struct.templ_non_tu = type { i32 (...)** }
%struct.templ_non_tu.0 = type { i32 (...)** }
%struct.templ_non_tu.1 = type { i32 (...)** }

@_ZTV6non_tu = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI6non_tu to i8*), i8* bitcast (void (%struct.non_tu*)* @_ZN6non_tu2f1Ev to i8*)] }, align 8
@v1 = dso_local global { { i8** } } { { i8** } { i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV6non_tu, i32 0, inrange i32 0, i32 2) } }, align 8, !dbg !0
@v5 = dso_local global %struct.ref_internal zeroinitializer, align 1, !dbg !5
@_ZTV12templ_non_tuIiE = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI12templ_non_tuIiE to i8*), i8* bitcast (void (%struct.templ_non_tu*)* @_ZN12templ_non_tuIiE2f1Ev to i8*)] }, align 8
@v2 = dso_local global { { i8** } } { { i8** } { i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV12templ_non_tuIiE, i32 0, inrange i32 0, i32 2) } }, align 8, !dbg !13
@_ZTV12templ_non_tuIlE = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI12templ_non_tuIlE to i8*), i8* bitcast (void (%struct.templ_non_tu.0*)* @_ZN12templ_non_tuIlE2f1Ev to i8*)] }, align 8
@v3 = dso_local global { { i8** } } { { i8** } { i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV12templ_non_tuIlE, i32 0, inrange i32 0, i32 2) } }, align 8, !dbg !32
@_ZTV12templ_non_tuIbE = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI12templ_non_tuIbE to i8*), i8* bitcast (void (%struct.templ_non_tu.1*)* @_ZN12templ_non_tuIbE2f1Ev to i8*)] }, align 8
@v4 = dso_local global { { i8** } } { { i8** } { i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV12templ_non_tuIbE, i32 0, inrange i32 0, i32 2) } }, align 8, !dbg !46
@v6 = dso_local global %class.ref_internal_template zeroinitializer, align 1, !dbg !60
@v7 = dso_local global %class.ref_from_ref_internal_template zeroinitializer, align 1, !dbg !69
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS6non_tu = dso_local constant [8 x i8] c"6non_tu\00", align 1
@_ZTI6non_tu = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6non_tu, i32 0, i32 0) }, align 8
@_ZTS12templ_non_tuIiE = dso_local constant [18 x i8] c"12templ_non_tuIiE\00", align 1
@_ZTI12templ_non_tuIiE = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @_ZTS12templ_non_tuIiE, i32 0, i32 0) }, align 8
@_ZTS12templ_non_tuIlE = dso_local constant [18 x i8] c"12templ_non_tuIlE\00", align 1
@_ZTI12templ_non_tuIlE = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @_ZTS12templ_non_tuIlE, i32 0, i32 0) }, align 8
@_ZTS12templ_non_tuIbE = dso_local constant [18 x i8] c"12templ_non_tuIbE\00", align 1
@_ZTI12templ_non_tuIbE = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @_ZTS12templ_non_tuIbE, i32 0, i32 0) }, align 8

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN6non_tu2f1Ev(%struct.non_tu* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !87 {
entry:
  %this.addr = alloca %struct.non_tu*, align 8
  store %struct.non_tu* %this, %struct.non_tu** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.non_tu** %this.addr, metadata !88, metadata !DIExpression()), !dbg !90
  %this1 = load %struct.non_tu*, %struct.non_tu** %this.addr, align 8
  ret void, !dbg !91
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIiE2f1Ev(%struct.templ_non_tu* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !92 {
entry:
  %this.addr = alloca %struct.templ_non_tu*, align 8
  store %struct.templ_non_tu* %this, %struct.templ_non_tu** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu** %this.addr, metadata !93, metadata !DIExpression()), !dbg !95
  %this1 = load %struct.templ_non_tu*, %struct.templ_non_tu** %this.addr, align 8
  ret void, !dbg !96
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIlE2f1Ev(%struct.templ_non_tu.0* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !97 {
entry:
  %this.addr = alloca %struct.templ_non_tu.0*, align 8
  store %struct.templ_non_tu.0* %this, %struct.templ_non_tu.0** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu.0** %this.addr, metadata !98, metadata !DIExpression()), !dbg !100
  %this1 = load %struct.templ_non_tu.0*, %struct.templ_non_tu.0** %this.addr, align 8
  ret void, !dbg !101
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIbE2f1Ev(%struct.templ_non_tu.1* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !102 {
entry:
  %this.addr = alloca %struct.templ_non_tu.1*, align 8
  store %struct.templ_non_tu.1* %this, %struct.templ_non_tu.1** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu.1** %this.addr, metadata !103, metadata !DIExpression()), !dbg !105
  %this1 = load %struct.templ_non_tu.1*, %struct.templ_non_tu.1** %this.addr, align 8
  ret void, !dbg !106
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!81, !82, !83, !84, !85}
!llvm.ident = !{!86}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v1", scope: !2, file: !3, line: 8, type: !71, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0 (git@github.com:llvm/llvm-project.git 862896df6210f6660514f9f68051cd0371832f37)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "84728cc618f91cd3bdecda0aa2d09c2d")
!4 = !{!0, !5, !13, !32, !46, !60, !69}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "v5", scope: !2, file: !3, line: 17, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ref_internal", file: !3, line: 14, size: 8, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS12ref_internal")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !7, file: !3, line: 15, baseType: !10, size: 8)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "internal", scope: !11, file: !3, line: 12, size: 8, flags: DIFlagTypePassByValue, elements: !12)
!11 = !DINamespace(scope: null)
!12 = !{}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "v2", scope: !2, file: !3, line: 33, type: !15, isLocal: false, isDefinition: true)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ref_templ_non_tu", file: !3, line: 30, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !16, identifier: "_ZTS16ref_templ_non_tu")
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !15, file: !3, line: 31, baseType: !18, size: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STNtempl_non_tu|<int>", file: !3, line: 26, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !19, vtableHolder: !18, templateParams: !30)
!19 = !{!20, !26}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$templ_non_tu", scope: !3, file: !3, baseType: !21, size: 64, flags: DIFlagArtificial)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !23, size: 64)
!23 = !DISubroutineType(types: !24)
!24 = !{!25}
!25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!26 = !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIiE2f1Ev", scope: !18, file: !3, line: 27, type: !27, scopeLine: 27, containingType: !18, virtualIndex: 0, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !29}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = !{!31}
!31 = !DITemplateTypeParameter(name: "T", type: !25)
!32 = !DIGlobalVariableExpression(var: !33, expr: !DIExpression())
!33 = distinct !DIGlobalVariable(name: "v3", scope: !2, file: !3, line: 45, type: !34, isLocal: false, isDefinition: true)
!34 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ref_templ_non_tu_simple", file: !3, line: 42, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !35, identifier: "_ZTS23ref_templ_non_tu_simple")
!35 = !{!36}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !34, file: !3, line: 43, baseType: !37, size: 64)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "templ_non_tu", file: !3, line: 38, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !38, vtableHolder: !37, templateParams: !43)
!38 = !{!20, !39}
!39 = !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIlE2f1Ev", scope: !37, file: !3, line: 39, type: !40, scopeLine: 39, containingType: !37, virtualIndex: 0, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !42}
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!43 = !{!44}
!44 = !DITemplateTypeParameter(name: "T", type: !45)
!45 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!46 = !DIGlobalVariableExpression(var: !47, expr: !DIExpression())
!47 = distinct !DIGlobalVariable(name: "v4", scope: !2, file: !3, line: 56, type: !48, isLocal: false, isDefinition: true)
!48 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ref_templ_non_tu_mangled", file: !3, line: 53, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !49, identifier: "_ZTS24ref_templ_non_tu_mangled")
!49 = !{!50}
!50 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !48, file: !3, line: 54, baseType: !51, size: 64)
!51 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_STNtempl_non_tu|<bool>", file: !3, line: 49, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !52, vtableHolder: !51, templateParams: !57)
!52 = !{!20, !53}
!53 = !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIbE2f1Ev", scope: !51, file: !3, line: 50, type: !54, scopeLine: 50, containingType: !51, virtualIndex: 0, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual)
!54 = !DISubroutineType(types: !55)
!55 = !{null, !56}
!56 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!57 = !{!58}
!58 = !DITemplateTypeParameter(name: "T", type: !59)
!59 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = distinct !DIGlobalVariable(name: "v6", scope: !2, file: !3, line: 66, type: !62, isLocal: false, isDefinition: true)
!62 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ref_internal_template", file: !3, line: 63, size: 8, flags: DIFlagTypePassByValue, elements: !63, identifier: "_ZTS21ref_internal_template")
!63 = !{!64}
!64 = !DIDerivedType(tag: DW_TAG_member, name: "ax", scope: !62, file: !3, line: 64, baseType: !65, size: 8)
!65 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "internal_template<ref_from_ref_internal_template>", scope: !11, file: !3, line: 59, size: 8, flags: DIFlagTypePassByValue, elements: !12, templateParams: !66)
!66 = !{!67}
!67 = !DITemplateTypeParameter(type: !68)
!68 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ref_from_ref_internal_template", file: !3, line: 61, size: 8, flags: DIFlagTypePassByValue, elements: !12, identifier: "_ZTS30ref_from_ref_internal_template")
!69 = !DIGlobalVariableExpression(var: !70, expr: !DIExpression())
!70 = distinct !DIGlobalVariable(name: "v7", scope: !2, file: !3, line: 67, type: !68, isLocal: false, isDefinition: true)
!71 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tu_ref_non_tu", file: !3, line: 5, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !72, identifier: "_ZTS13tu_ref_non_tu")
!72 = !{!73}
!73 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !71, file: !3, line: 6, baseType: !74, size: 64)
!74 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "non_tu", file: !3, line: 1, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !75, vtableHolder: !74)
!75 = !{!76, !77}
!76 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$non_tu", scope: !3, file: !3, baseType: !21, size: 64, flags: DIFlagArtificial)
!77 = !DISubprogram(name: "f1", linkageName: "_ZN6non_tu2f1Ev", scope: !74, file: !3, line: 2, type: !78, scopeLine: 2, containingType: !74, virtualIndex: 0, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual)
!78 = !DISubroutineType(types: !79)
!79 = !{null, !80}
!80 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !74, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!81 = !{i32 7, !"Dwarf Version", i32 5}
!82 = !{i32 2, !"Debug Info Version", i32 3}
!83 = !{i32 1, !"wchar_size", i32 4}
!84 = !{i32 7, !"uwtable", i32 2}
!85 = !{i32 7, !"frame-pointer", i32 2}
!86 = !{!"clang version 15.0.0 (git@github.com:llvm/llvm-project.git 862896df6210f6660514f9f68051cd0371832f37)"}
!87 = distinct !DISubprogram(name: "f1", linkageName: "_ZN6non_tu2f1Ev", scope: !74, file: !3, line: 4, type: !78, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !77, retainedNodes: !12)
!88 = !DILocalVariable(name: "this", arg: 1, scope: !87, type: !89, flags: DIFlagArtificial | DIFlagObjectPointer)
!89 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !74, size: 64)
!90 = !DILocation(line: 0, scope: !87)
!91 = !DILocation(line: 4, column: 20, scope: !87)
!92 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIiE2f1Ev", scope: !18, file: !3, line: 29, type: !27, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !26, retainedNodes: !12)
!93 = !DILocalVariable(name: "this", arg: 1, scope: !92, type: !94, flags: DIFlagArtificial | DIFlagObjectPointer)
!94 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!95 = !DILocation(line: 0, scope: !92)
!96 = !DILocation(line: 29, column: 31, scope: !92)
!97 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIlE2f1Ev", scope: !37, file: !3, line: 41, type: !40, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !39, retainedNodes: !12)
!98 = !DILocalVariable(name: "this", arg: 1, scope: !97, type: !99, flags: DIFlagArtificial | DIFlagObjectPointer)
!99 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!100 = !DILocation(line: 0, scope: !97)
!101 = !DILocation(line: 41, column: 32, scope: !97)
!102 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIbE2f1Ev", scope: !51, file: !3, line: 52, type: !54, scopeLine: 52, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !53, retainedNodes: !12)
!103 = !DILocalVariable(name: "this", arg: 1, scope: !102, type: !104, flags: DIFlagArtificial | DIFlagObjectPointer)
!104 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64)
!105 = !DILocation(line: 0, scope: !102)
!106 = !DILocation(line: 52, column: 32, scope: !102)
