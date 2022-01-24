; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; Test that a type unit referencing a non-type unit produces a declaration of
; the referent in the referee.

; Also check that an attempt to reference an internal linkage (defined in an anonymous
; namespace) type from a type unit (could happen with a pimpl idiom, for instance -
; it does mean the linkage-having type can only be defined in one translation
; unit anyway) forces the referent to not be placed in a type unit (because the
; declaration of the internal linkage type would be ambiguous/wouldn't allow a
; consumer to find the definition with certainty)

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
; // Modify templ_non_tu<long>'s name to be mangled ('_STN' name '|' args)
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
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"tu_ref_non_tu"

; CHECK-LABEL: Type Unit:
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration       (true)
; CHECK-NEXT: DW_AT_name      ("templ_non_tu<int>")

; CHECK-LABEL: Type Unit:
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu_simple"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration       (true)
; CHECK-NEXT: DW_AT_name      ("templ_non_tu")
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type    {{.*}}"long"

; CHECK-LABEL: Type Unit:
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}"ref_templ_non_tu_mangled"
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration       (true)
; CHECK-NEXT: DW_AT_name      ("_STNtempl_non_tu|<bool>")
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type    {{.*}}"bool"


; CHECK-LABEL: Compile Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT:   DW_AT_declaration       (true)
; CHECK-NEXT:   DW_AT_signature (0xb1cde890d320f5c2)

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

%struct.ref_internal = type { %"struct.(anonymous namespace)::internal" }
%"struct.(anonymous namespace)::internal" = type { i8 }
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
define dso_local void @_ZN6non_tu2f1Ev(%struct.non_tu* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !76 {
entry:
  %this.addr = alloca %struct.non_tu*, align 8
  store %struct.non_tu* %this, %struct.non_tu** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.non_tu** %this.addr, metadata !77, metadata !DIExpression()), !dbg !79
  %this1 = load %struct.non_tu*, %struct.non_tu** %this.addr, align 8
  ret void, !dbg !80
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIiE2f1Ev(%struct.templ_non_tu* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !81 {
entry:
  %this.addr = alloca %struct.templ_non_tu*, align 8
  store %struct.templ_non_tu* %this, %struct.templ_non_tu** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu** %this.addr, metadata !82, metadata !DIExpression()), !dbg !84
  %this1 = load %struct.templ_non_tu*, %struct.templ_non_tu** %this.addr, align 8
  ret void, !dbg !85
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIlE2f1Ev(%struct.templ_non_tu.0* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !86 {
entry:
  %this.addr = alloca %struct.templ_non_tu.0*, align 8
  store %struct.templ_non_tu.0* %this, %struct.templ_non_tu.0** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu.0** %this.addr, metadata !87, metadata !DIExpression()), !dbg !89
  %this1 = load %struct.templ_non_tu.0*, %struct.templ_non_tu.0** %this.addr, align 8
  ret void, !dbg !90
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_ZN12templ_non_tuIbE2f1Ev(%struct.templ_non_tu.1* noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 !dbg !91 {
entry:
  %this.addr = alloca %struct.templ_non_tu.1*, align 8
  store %struct.templ_non_tu.1* %this, %struct.templ_non_tu.1** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.templ_non_tu.1** %this.addr, metadata !92, metadata !DIExpression()), !dbg !94
  %this1 = load %struct.templ_non_tu.1*, %struct.templ_non_tu.1** %this.addr, align 8
  ret void, !dbg !95
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!70, !71, !72, !73, !74}
!llvm.ident = !{!75}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v1", scope: !2, file: !3, line: 8, type: !60, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0 (git@github.com:llvm/llvm-project.git ab4756338c5b2216d52d9152b2f7e65f233c4dac)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "decl.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!4 = !{!0, !5, !13, !32, !46}
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
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "templ_non_tu<int>", file: !3, line: 26, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !19, vtableHolder: !18, templateParams: !30)
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
!60 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tu_ref_non_tu", file: !3, line: 5, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !61, identifier: "_ZTS13tu_ref_non_tu")
!61 = !{!62}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "v1", scope: !60, file: !3, line: 6, baseType: !63, size: 64)
!63 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "non_tu", file: !3, line: 1, size: 64, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !64, vtableHolder: !63)
!64 = !{!65, !66}
!65 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$non_tu", scope: !3, file: !3, baseType: !21, size: 64, flags: DIFlagArtificial)
!66 = !DISubprogram(name: "f1", linkageName: "_ZN6non_tu2f1Ev", scope: !63, file: !3, line: 2, type: !67, scopeLine: 2, containingType: !63, virtualIndex: 0, flags: DIFlagPrototyped, spFlags: DISPFlagVirtual)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !69}
!69 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !63, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!70 = !{i32 7, !"Dwarf Version", i32 5}
!71 = !{i32 2, !"Debug Info Version", i32 3}
!72 = !{i32 1, !"wchar_size", i32 4}
!73 = !{i32 7, !"uwtable", i32 1}
!74 = !{i32 7, !"frame-pointer", i32 2}
!75 = !{!"clang version 14.0.0 (git@github.com:llvm/llvm-project.git ab4756338c5b2216d52d9152b2f7e65f233c4dac)"}
!76 = distinct !DISubprogram(name: "f1", linkageName: "_ZN6non_tu2f1Ev", scope: !63, file: !3, line: 4, type: !67, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !66, retainedNodes: !12)
!77 = !DILocalVariable(name: "this", arg: 1, scope: !76, type: !78, flags: DIFlagArtificial | DIFlagObjectPointer)
!78 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !63, size: 64)
!79 = !DILocation(line: 0, scope: !76)
!80 = !DILocation(line: 4, column: 20, scope: !76)
!81 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIiE2f1Ev", scope: !18, file: !3, line: 29, type: !27, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !26, retainedNodes: !12)
!82 = !DILocalVariable(name: "this", arg: 1, scope: !81, type: !83, flags: DIFlagArtificial | DIFlagObjectPointer)
!83 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!84 = !DILocation(line: 0, scope: !81)
!85 = !DILocation(line: 29, column: 31, scope: !81)
!86 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIlE2f1Ev", scope: !37, file: !3, line: 41, type: !40, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !39, retainedNodes: !12)
!87 = !DILocalVariable(name: "this", arg: 1, scope: !86, type: !88, flags: DIFlagArtificial | DIFlagObjectPointer)
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!89 = !DILocation(line: 0, scope: !86)
!90 = !DILocation(line: 41, column: 32, scope: !86)
!91 = distinct !DISubprogram(name: "f1", linkageName: "_ZN12templ_non_tuIbE2f1Ev", scope: !51, file: !3, line: 52, type: !54, scopeLine: 52, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !53, retainedNodes: !12)
!92 = !DILocalVariable(name: "this", arg: 1, scope: !91, type: !93, flags: DIFlagArtificial | DIFlagObjectPointer)
!93 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64)
!94 = !DILocation(line: 0, scope: !91)
!95 = !DILocation(line: 52, column: 32, scope: !91)
