; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj < %s | llvm-dwarfdump --debug-info --debug-gnu-pubnames --debug-gnu-pubtypes - | FileCheck %s --implicit-check-not "{{DW_TAG|NULL}}"
; ModuleID = 'dwarf-public-names.cpp'
;
; Generated from:
;
; struct C {
;   void member_function();
;   static int static_member_function();
;   static int static_member_variable;
; };
;
; int C::static_member_variable = 0;
;
; void C::member_function() {
;   static_member_variable = 0;
; }
;
; int C::static_member_function() {
;   return static_member_variable;
; }
;
; C global_variable;
;
; int global_function() {
;   return -1;
; }
;
; namespace ns {
;   void global_namespace_function() {
;     global_variable.member_function();
;   }
;   int global_namespace_variable = 1;
;   extern int global_namespace_variable_decl;
;   struct D {
;     int A;
;   } d;
; }
;
; using ns::global_namespace_variable_decl;
;
; namespace {
; int i;
; }
;
; int *f3() {
;   static int z;
;   return &z;
; }
;
; namespace {
; namespace inner {
; int b;
; }
; }
;
; namespace outer {
; namespace {
; int c;
; }
; }
;
; int f7() {
;   return i + *f3() + inner::b + outer::c;
; }

; ASM: .section        .debug_gnu_pubnames
; ASM: .byte   32                      # Attributes: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "C::static_member_variable"           # External Name
; ASM: .byte   32                      # Attributes: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_variable"       # External Name
; ASM: .byte   32                      # Attributes: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "ns::global_namespace_variable"       # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Attributes: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_GNU_pubnames (true)
; CHECK-NOT: DW_AT_GNU_pubtypes [

; CHECK:   [[STATIC_MEM_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK:     DW_AT_specification {{.*}} "static_member_variable"

; CHECK:   [[C:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK:     DW_AT_name ("C")
; CHECK:     DW_TAG_member
; CHECK:       DW_AT_name ("static_member_variable")
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_linkage_name
; CHECK:       DW_AT_name ("member_function")
; CHECK:       DW_TAG_formal_parameter
; CHECK:       NULL
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_linkage_name
; CHECK:       DW_AT_name ("static_member_function")
; CHECK:     NULL

; CHECK:   [[INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK:     DW_AT_name ("int")
; CHECK:   DW_TAG_pointer_type

; CHECK:   [[GLOB_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK:     DW_AT_name ("global_variable")

; CHECK:   [[NS:0x[0-9a-f]+]]: DW_TAG_namespace
; CHECK:     DW_AT_name ("ns")
; CHECK:     [[GLOB_NS_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK:       DW_AT_name ("global_namespace_variable")
; CHECK-NOT:   DW_AT_specification
; CHECK:       DW_AT_location
; CHECK-NOT:   DW_AT_specification
; CHECK:     [[D_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK:       DW_AT_name ("d")
; CHECK-NOT:   DW_AT_specification
; CHECK:       DW_AT_location
; CHECK-NOT:   DW_AT_specification
; CHECK:     [[D:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK:       DW_AT_name ("D")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     [[GLOB_NS_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:       DW_AT_linkage_name
; CHECK:       DW_AT_name ("global_namespace_function")
; CHECK:     DW_TAG_variable
; CHECK:     NULL

; CHECK:   [[F3:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_name ("f3")
; CHECK:     [[F3_Z:.*]]: DW_TAG_variable
; CHECK:       DW_AT_name ("z")
; CHECK:       DW_AT_location
; CHECK:     NULL

; CHECK:   [[ANON:.*]]: DW_TAG_namespace
; CHECK-NOT: DW_AT_name
; CHECK:     [[ANON_I:.*]]: DW_TAG_variable
; CHECK:       DW_AT_name ("i")
; CHECK:       [[ANON_INNER:.*]]:  DW_TAG_namespace
; CHECK:         DW_AT_name ("inner")
; CHECK:         [[ANON_INNER_B:.*]]: DW_TAG_variable
; CHECK:           DW_AT_name ("b")
; CHECK:         NULL
; CHECK:     NULL

; CHECK:   [[OUTER:.*]]: DW_TAG_namespace
; CHECK:     DW_AT_name ("outer")
; CHECK:     [[OUTER_ANON:.*]]:  DW_TAG_namespace
; CHECK-NOT:   DW_AT_name
; CHECK:       [[OUTER_ANON_C:.*]]: DW_TAG_variable
; CHECK:         DW_AT_name ("c")
; CHECK:       NULL
; CHECK:     NULL

; CHECK:   DW_TAG_enumeration
; CHECK:     [[UNNAMED_ENUM_ENUMERATOR:0x[0-9a-f]+]]:  DW_TAG_enumerator
; CHECK:       DW_AT_name ("unnamed_enum_enumerator")
; CHECK:     NULL

; CHECK:   [[UNSIGNED_INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK:     DW_AT_name ("unsigned int")

; CHECK:   [[NAMED_ENUM:0x[0-9a-f]+]]: DW_TAG_enumeration
; CHECK:     DW_AT_name ("named_enum")
; CHECK:     [[NAMED_ENUM_ENUMERATOR:0x[0-9a-f]+]]:  DW_TAG_enumerator
; CHECK:       DW_AT_name ("named_enum_enumerator")
; CHECK:     NULL

; CHECK:   [[NAMED_ENUM_CLASS:0x[0-9a-f]+]]: DW_TAG_enumeration
; CHECK:     DW_AT_name ("named_enum_class")
; CHECK:     [[NAMED_ENUM_CLASS_ENUMERATOR:0x[0-9a-f]+]]:  DW_TAG_enumerator
; CHECK:       DW_AT_name ("named_enum_class_enumerator")
; CHECK:     NULL

; CHECK:   [[MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_specification {{.*}} "_ZN1C15member_functionEv"
; CHECK:     DW_TAG_formal_parameter
; CHECK:     NULL

; CHECK:   [[STATIC_MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_specification {{.*}} "_ZN1C22static_member_functionEv"

; CHECK:   [[GLOBAL_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name
; CHECK:     DW_AT_name ("global_function")

; CHECK:   [[GLOBAL_F7:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name
; CHECK:     DW_AT_name ("f7")

; CHECK:   DW_TAG_imported_declaration
; CHECK:   DW_TAG_pointer_type
; CHECK:   DW_TAG_pointer_type
; CHECK:   NULL

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = {{.*}}, version = 0x0002, unit_offset = 0x00000000, unit_size = {{.*}}
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-NEXT:  [[STATIC_MEM_VAR]] EXTERNAL VARIABLE "C::static_member_variable"
; CHECK-NEXT:  [[GLOB_VAR]] EXTERNAL VARIABLE "global_variable"
; CHECK-NEXT:  [[NS]] EXTERNAL TYPE     "ns"
; CHECK-NEXT:  [[GLOB_NS_VAR]] EXTERNAL VARIABLE "ns::global_namespace_variable"
; CHECK-NEXT:  [[D_VAR]] EXTERNAL VARIABLE "ns::d"
; CHECK-NEXT:  [[GLOB_NS_FUNC]] EXTERNAL FUNCTION "ns::global_namespace_function"
; CHECK-NEXT:  [[F3]] EXTERNAL FUNCTION "f3"
; GCC Doesn't put local statics in pubnames, but it seems not unreasonable and
; comes out naturally from LLVM's implementation, so I'm OK with it for now. If
; it's demonstrated that this is a major size concern or degrades debug info
; consumer behavior, feel free to change it.
; CHECK-NEXT:  [[F3_Z]] STATIC VARIABLE "f3::z"
; CHECK-NEXT:  [[ANON]] EXTERNAL TYPE "(anonymous namespace)"
; CHECK-NEXT:  [[ANON_I]] STATIC VARIABLE "(anonymous namespace)::i"
; CHECK-NEXT:  [[ANON_INNER]] EXTERNAL TYPE "(anonymous namespace)::inner"
; CHECK-NEXT:  [[ANON_INNER_B]] STATIC VARIABLE "(anonymous namespace)::inner::b"
; CHECK-NEXT:  [[OUTER]] EXTERNAL TYPE "outer"
; CHECK-NEXT:  [[OUTER_ANON]] EXTERNAL TYPE "outer::(anonymous namespace)"
; CHECK-NEXT:  [[OUTER_ANON_C]] STATIC VARIABLE "outer::(anonymous namespace)::c"
; CHECK-NEXT:  [[UNNAMED_ENUM_ENUMERATOR]] STATIC VARIABLE  "unnamed_enum_enumerator"
; CHECK-NEXT:  [[NAMED_ENUM_ENUMERATOR]] STATIC VARIABLE  "named_enum_enumerator"
; CHECK-NEXT:  [[NAMED_ENUM_CLASS_ENUMERATOR]] STATIC VARIABLE  "named_enum_class_enumerator"
; CHECK-NEXT:  [[MEM_FUNC]] EXTERNAL FUNCTION "C::member_function"
; CHECK-NEXT:  [[STATIC_MEM_FUNC]] EXTERNAL FUNCTION "C::static_member_function"
; FIXME: GCC produces enumerators as EXTERNAL, not STATIC
; CHECK-NEXT:  [[GLOBAL_FUNC]] EXTERNAL FUNCTION "global_function"
; CHECK-NEXT:  [[GLOBAL_F7]] EXTERNAL FUNCTION "f7"

; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK: Offset     Linkage  Kind     Name
; CHECK-NEXT:  [[C]] EXTERNAL TYPE     "C"
; CHECK-NEXT:  [[INT]] STATIC   TYPE     "int"
; CHECK-NEXT:  [[D]] EXTERNAL TYPE     "ns::D"
; CHECK-NEXT:  [[UNSIGNED_INT]] STATIC   TYPE     "unsigned int"
; CHECK-NEXT:  [[NAMED_ENUM]] EXTERNAL TYPE     "named_enum"
; CHECK-NEXT:  [[NAMED_ENUM_CLASS]] EXTERNAL TYPE     "named_enum_class"

%struct.C = type { i8 }
%"struct.ns::D" = type { i32 }

@_ZN1C22static_member_variableE = dso_local global i32 0, align 4, !dbg !0
@global_variable = dso_local global %struct.C zeroinitializer, align 1, !dbg !18
@_ZN2ns25global_namespace_variableE = dso_local global i32 1, align 4, !dbg !29
@_ZN2ns1dE = dso_local global %"struct.ns::D" zeroinitializer, align 4, !dbg !32
@_ZZ2f3vE1z = internal global i32 0, align 4, !dbg !37
@_ZN12_GLOBAL__N_11iE = internal global i32 0, align 4, !dbg !44
@_ZN12_GLOBAL__N_15inner1bE = internal global i32 0, align 4, !dbg !47
@_ZN5outer12_GLOBAL__N_11cE = internal global i32 0, align 4, !dbg !50

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_ZN1C15member_functionEv(ptr %this) #0 align 2 !dbg !61 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !62, metadata !DIExpression()), !dbg !64
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 0, ptr @_ZN1C22static_member_variableE, align 4, !dbg !65
  ret void, !dbg !66
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_ZN1C22static_member_functionEv() #0 align 2 !dbg !67 {
entry:
  %0 = load i32, ptr @_ZN1C22static_member_variableE, align 4, !dbg !68
  ret i32 %0, !dbg !69
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z15global_functionv() #0 !dbg !70 {
entry:
  ret i32 -1, !dbg !71
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_ZN2ns25global_namespace_functionEv() #0 !dbg !72 {
entry:
  call void @_ZN1C15member_functionEv(ptr @global_variable), !dbg !75
  ret void, !dbg !76
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local ptr @_Z2f3v() #0 !dbg !39 {
entry:
  ret ptr @_ZZ2f3vE1z, !dbg !77
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z2f7v() #0 !dbg !78 {
entry:
  %0 = load i32, ptr @_ZN12_GLOBAL__N_11iE, align 4, !dbg !79
  %call = call ptr @_Z2f3v(), !dbg !80
  %1 = load i32, ptr %call, align 4, !dbg !81
  %add = add nsw i32 %0, %1, !dbg !82
  %2 = load i32, ptr @_ZN12_GLOBAL__N_15inner1bE, align 4, !dbg !83
  %add1 = add nsw i32 %add, %2, !dbg !84
  %3 = load i32, ptr @_ZN5outer12_GLOBAL__N_11cE, align 4, !dbg !85
  %add2 = add nsw i32 %add1, %3, !dbg !86
  %add3 = add nsw i32 %add2, 0, !dbg !87
  %add4 = add nsw i32 %add3, 0, !dbg !88
  %add5 = add nsw i32 %add4, 0, !dbg !89
  ret i32 %add5, !dbg !90
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!57, !58, !59}
!llvm.ident = !{!60}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "static_member_variable", linkageName: "_ZN1C22static_member_variableE", scope: !2, file: !3, line: 7, type: !13, isLocal: false, isDefinition: true, declaration: !22)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_03, file: !3, producer: "clang version 9.0.0 (trunk 363288) (llvm/trunk 363294)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !16, globals: !17, imports: !54, nameTableKind: GNU)
!3 = !DIFile(filename: "names.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!4 = !{!5, !9, !12}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !3, line: 49, baseType: !6, size: 32, elements: !7)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{!8}
!8 = !DIEnumerator(name: "unnamed_enum_enumerator", value: 0, isUnsigned: true)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "named_enum", file: !3, line: 52, baseType: !6, size: 32, elements: !10, identifier: "_ZTS10named_enum")
!10 = !{!11}
!11 = !DIEnumerator(name: "named_enum_enumerator", value: 0, isUnsigned: true)
!12 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "named_enum_class", file: !3, line: 55, baseType: !13, size: 32, flags: DIFlagEnumClass, elements: !14, identifier: "_ZTS16named_enum_class")
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DIEnumerator(name: "named_enum_class_enumerator", value: 0)
!16 = !{!13}
!17 = !{!0, !18, !29, !32, !37, !44, !47, !50}
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "global_variable", scope: !2, file: !3, line: 13, type: !20, isLocal: false, isDefinition: true)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS1C")
!21 = !{!22, !23, !27}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "static_member_variable", scope: !20, file: !3, line: 4, baseType: !13, flags: DIFlagStaticMember)
!23 = !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: !20, file: !3, line: 2, type: !24, scopeLine: 2, flags: DIFlagPrototyped, spFlags: 0)
!24 = !DISubroutineType(types: !25)
!25 = !{null, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: !20, file: !3, line: 3, type: !28, scopeLine: 3, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!28 = !DISubroutineType(types: !16)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "global_namespace_variable", linkageName: "_ZN2ns25global_namespace_variableE", scope: !31, file: !3, line: 19, type: !13, isLocal: false, isDefinition: true)
!31 = !DINamespace(name: "ns", scope: null)
!32 = !DIGlobalVariableExpression(var: !33, expr: !DIExpression())
!33 = distinct !DIGlobalVariable(name: "d", linkageName: "_ZN2ns1dE", scope: !31, file: !3, line: 23, type: !34, isLocal: false, isDefinition: true)
!34 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !31, file: !3, line: 21, size: 32, flags: DIFlagTypePassByValue, elements: !35, identifier: "_ZTSN2ns1DE")
!35 = !{!36}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !34, file: !3, line: 22, baseType: !13, size: 32)
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "z", scope: !39, file: !3, line: 33, type: !13, isLocal: true, isDefinition: true)
!39 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !3, file: !3, line: 32, type: !40, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
!40 = !DISubroutineType(types: !41)
!41 = !{!42}
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!43 = !{}
!44 = !DIGlobalVariableExpression(var: !45, expr: !DIExpression())
!45 = distinct !DIGlobalVariable(name: "i", linkageName: "_ZN12_GLOBAL__N_11iE", scope: !46, file: !3, line: 29, type: !13, isLocal: true, isDefinition: true)
!46 = !DINamespace(scope: null)
!47 = !DIGlobalVariableExpression(var: !48, expr: !DIExpression())
!48 = distinct !DIGlobalVariable(name: "b", linkageName: "_ZN12_GLOBAL__N_15inner1bE", scope: !49, file: !3, line: 39, type: !13, isLocal: true, isDefinition: true)
!49 = !DINamespace(name: "inner", scope: !46)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = distinct !DIGlobalVariable(name: "c", linkageName: "_ZN5outer12_GLOBAL__N_11cE", scope: !52, file: !3, line: 45, type: !13, isLocal: true, isDefinition: true)
!52 = !DINamespace(scope: !53)
!53 = !DINamespace(name: "outer", scope: null)
!54 = !{!55}
!55 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !56, file: !3, line: 26)
!56 = !DIGlobalVariable(name: "global_namespace_variable_decl", linkageName: "_ZN2ns30global_namespace_variable_declE", scope: !31, file: !3, line: 20, type: !13, isLocal: false, isDefinition: false)
!57 = !{i32 2, !"Dwarf Version", i32 4}
!58 = !{i32 2, !"Debug Info Version", i32 3}
!59 = !{i32 1, !"wchar_size", i32 4}
!60 = !{!"clang version 9.0.0 (trunk 363288) (llvm/trunk 363294)"}
!61 = distinct !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: !20, file: !3, line: 9, type: !24, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !23, retainedNodes: !43)
!62 = !DILocalVariable(name: "this", arg: 1, scope: !61, type: !63, flags: DIFlagArtificial | DIFlagObjectPointer)
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!64 = !DILocation(line: 0, scope: !61)
!65 = !DILocation(line: 9, column: 52, scope: !61)
!66 = !DILocation(line: 9, column: 57, scope: !61)
!67 = distinct !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: !20, file: !3, line: 11, type: !28, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !27, retainedNodes: !43)
!68 = !DILocation(line: 11, column: 42, scope: !67)
!69 = !DILocation(line: 11, column: 35, scope: !67)
!70 = distinct !DISubprogram(name: "global_function", linkageName: "_Z15global_functionv", scope: !3, file: !3, line: 15, type: !28, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
!71 = !DILocation(line: 15, column: 25, scope: !70)
!72 = distinct !DISubprogram(name: "global_namespace_function", linkageName: "_ZN2ns25global_namespace_functionEv", scope: !31, file: !3, line: 18, type: !73, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
!73 = !DISubroutineType(types: !74)
!74 = !{null}
!75 = !DILocation(line: 18, column: 52, scope: !72)
!76 = !DILocation(line: 18, column: 71, scope: !72)
!77 = !DILocation(line: 34, column: 3, scope: !39)
!78 = distinct !DISubprogram(name: "f7", linkageName: "_Z2f7v", scope: !3, file: !3, line: 58, type: !28, scopeLine: 58, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
!79 = !DILocation(line: 59, column: 10, scope: !78)
!80 = !DILocation(line: 59, column: 15, scope: !78)
!81 = !DILocation(line: 59, column: 14, scope: !78)
!82 = !DILocation(line: 59, column: 12, scope: !78)
!83 = !DILocation(line: 59, column: 22, scope: !78)
!84 = !DILocation(line: 59, column: 20, scope: !78)
!85 = !DILocation(line: 59, column: 33, scope: !78)
!86 = !DILocation(line: 59, column: 31, scope: !78)
!87 = !DILocation(line: 59, column: 42, scope: !78)
!88 = !DILocation(line: 59, column: 68, scope: !78)
!89 = !DILocation(line: 60, column: 32, scope: !78)
!90 = !DILocation(line: 59, column: 3, scope: !78)
