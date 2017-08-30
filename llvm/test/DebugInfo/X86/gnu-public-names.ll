; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
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
; ASM: .byte   32                      # Kind: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_variable"       # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Kind: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: Compile Unit:
; CHECK: DW_AT_GNU_pubnames [DW_FORM_flag_present]   (true)
; CHECK-NOT: DW_AT_GNU_pubtypes [

; CHECK: [[STATIC_MEM_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "static_member_variable"

; CHECK: [[C:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_variable"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"

; CHECK: [[INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"

; CHECK: [[GLOB_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"

; CHECK: [[NS:0x[0-9a-f]+]]: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"

; CHECK: [[GLOB_NS_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "d"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "D"

; CHECK: [[GLOB_NS_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_namespace_function"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "f3"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[F3_Z:.*]]:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "z"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[ANON:.*]]: DW_TAG_namespace
; CHECK-NOT:   DW_AT_name
; CHECK: [[ANON_I:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER:.*]]:  DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "inner"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER_B:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[OUTER:.*]]: DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "outer"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[OUTER_ANON:.*]]:  DW_TAG_namespace
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK-NOT:     DW_AT_name
; CHECK: [[OUTER_ANON_C:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "c"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: We probably shouldn't bother describing the implicit
; import of the preceding anonymous namespace. This should be fixed
; in clang.
; CHECK:     DW_TAG_imported_module
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: DW_TAG_imported_declaration
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C15member_functionEv"

; CHECK: [[STATIC_MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C22static_member_functionEv"

; CHECK: [[GLOBAL_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_function"

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = {{.*}} version = 0x0002 unit_offset = 0x00000000 unit_size = {{.*}}
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-NEXT:  [[GLOBAL_FUNC]] EXTERNAL FUNCTION "global_function"
; CHECK-NEXT:  [[NS]] EXTERNAL TYPE     "ns"
; CHECK-NEXT:  [[OUTER_ANON_C]] STATIC VARIABLE "outer::(anonymous namespace)::c"
; CHECK-NEXT:  [[ANON_I]] STATIC VARIABLE "(anonymous namespace)::i"
; GCC Doesn't put local statics in pubnames, but it seems not unreasonable and
; comes out naturally from LLVM's implementation, so I'm OK with it for now. If
; it's demonstrated that this is a major size concern or degrades debug info
; consumer behavior, feel free to change it.
; CHECK-NEXT:  [[F3_Z]] STATIC VARIABLE "f3::z"
; CHECK-NEXT:  [[ANON]] EXTERNAL TYPE "(anonymous namespace)"
; CHECK-NEXT:  [[OUTER_ANON]] EXTERNAL TYPE "outer::(anonymous namespace)"
; CHECK-NEXT:  [[ANON_INNER_B]] STATIC VARIABLE "(anonymous namespace)::inner::b"
; CHECK-NEXT:  [[OUTER]] EXTERNAL TYPE "outer"
; CHECK-NEXT:  [[MEM_FUNC]] EXTERNAL FUNCTION "C::member_function"
; CHECK-NEXT:  [[GLOB_VAR]] EXTERNAL VARIABLE "global_variable"
; CHECK-NEXT:  [[GLOB_NS_VAR]] EXTERNAL VARIABLE "ns::global_namespace_variable"
; CHECK-NEXT:  [[ANON_INNER]] EXTERNAL TYPE "(anonymous namespace)::inner"
; CHECK-NEXT:  [[D_VAR]] EXTERNAL VARIABLE "ns::d"
; CHECK-NEXT:  [[GLOB_NS_FUNC]] EXTERNAL FUNCTION "ns::global_namespace_function"
; CHECK-NEXT:  [[STATIC_MEM_VAR]] EXTERNAL VARIABLE "C::static_member_variable"
; CHECK-NEXT:  [[STATIC_MEM_FUNC]] EXTERNAL FUNCTION "C::static_member_function"

; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[C]] EXTERNAL TYPE     "C"
; CHECK-DAG:  [[D]] EXTERNAL TYPE     "ns::D"
; CHECK-DAG:  [[INT]] STATIC   TYPE     "int"

source_filename = "test/DebugInfo/X86/gnu-public-names.ll"

%struct.C = type { i8 }
%"struct.ns::D" = type { i32 }

@_ZN1C22static_member_variableE = global i32 0, align 4, !dbg !0
@global_variable = global %struct.C zeroinitializer, align 1, !dbg !22
@_ZN2ns25global_namespace_variableE = global i32 1, align 4, !dbg !24
@_ZN2ns1dE = global %"struct.ns::D" zeroinitializer, align 4, !dbg !26
@_ZZ2f3vE1z = internal global i32 0, align 4, !dbg !28
@_ZN12_GLOBAL__N_11iE = internal global i32 0, align 4, !dbg !34
@_ZN12_GLOBAL__N_15inner1bE = internal global i32 0, align 4, !dbg !37
@_ZN5outer12_GLOBAL__N_11cE = internal global i32 0, align 4, !dbg !40

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 !dbg !51 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !52, metadata !54), !dbg !55
  %this1 = load %struct.C*, %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !56
  ret void, !dbg !57
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 !dbg !58 {
entry:
  %0 = load i32, i32* @_ZN1C22static_member_variableE, align 4, !dbg !59
  ret i32 %0, !dbg !59
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 !dbg !60 {
entry:
  ret i32 -1, !dbg !61
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 !dbg !62 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !65
  ret void, !dbg !66
}

; Function Attrs: nounwind uwtable
define i32* @_Z2f3v() #0 !dbg !30 {
entry:
  ret i32* @_ZZ2f3vE1z, !dbg !67
}

; Function Attrs: nounwind uwtable
define i32 @_Z2f7v() #0 !dbg !68 {
entry:
  %0 = load i32, i32* @_ZN12_GLOBAL__N_11iE, align 4, !dbg !69
  %call = call i32* @_Z2f3v(), !dbg !69
  %1 = load i32, i32* %call, align 4, !dbg !69
  %add = add nsw i32 %0, %1, !dbg !69
  %2 = load i32, i32* @_ZN12_GLOBAL__N_15inner1bE, align 4, !dbg !69
  %add1 = add nsw i32 %add, %2, !dbg !69
  %3 = load i32, i32* @_ZN5outer12_GLOBAL__N_11cE, align 4, !dbg !69
  %add2 = add nsw i32 %add1, %3, !dbg !69
  ret i32 %add2, !dbg !69
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!48, !49}
!llvm.ident = !{!50}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "static_member_variable", linkageName: "_ZN1C22static_member_variableE", scope: !2, file: !3, line: 7, type: !9, isLocal: false, isDefinition: true, declaration: !8)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.7.0 (trunk 234897) (llvm/trunk 234911)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !21, imports: !44)
!3 = !DIFile(filename: "gnu-public-names.cpp", directory: "/tmp/dbginfo")
!4 = !{}
!5 = !{!6, !17}
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 1, size: 8, align: 8, elements: !7, identifier: "_ZTS1C")
!7 = !{!8, !10, !14}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "static_member_variable", scope: !6, file: !3, line: 4, baseType: !9, flags: DIFlagStaticMember)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: !6, file: !3, line: 2, type: !11, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!14 = !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: !6, file: !3, line: 3, type: !15, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !18, file: !3, line: 29, size: 32, align: 32, elements: !19, identifier: "_ZTSN2ns1DE")
!18 = !DINamespace(name: "ns", scope: null)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !17, file: !3, line: 30, baseType: !9, size: 32, align: 32)
!21 = !{!0, !22, !24, !26, !28, !34, !37, !40}
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = !DIGlobalVariable(name: "global_variable", scope: !2, file: !3, line: 17, type: !6, isLocal: false, isDefinition: true)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = !DIGlobalVariable(name: "global_namespace_variable", linkageName: "_ZN2ns25global_namespace_variableE", scope: !18, file: !3, line: 27, type: !9, isLocal: false, isDefinition: true)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = !DIGlobalVariable(name: "d", linkageName: "_ZN2ns1dE", scope: !18, file: !3, line: 31, type: !17, isLocal: false, isDefinition: true)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = !DIGlobalVariable(name: "z", scope: !30, file: !3, line: 41, type: !9, isLocal: true, isDefinition: true)
!30 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !3, file: !3, line: 40, type: !31, isLocal: false, isDefinition: true, scopeLine: 40, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!31 = !DISubroutineType(types: !32)
!32 = !{!33}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = !DIGlobalVariable(name: "i", linkageName: "_ZN12_GLOBAL__N_11iE", scope: !36, file: !3, line: 37, type: !9, isLocal: true, isDefinition: true)
!36 = !DINamespace(scope: null)
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = !DIGlobalVariable(name: "b", linkageName: "_ZN12_GLOBAL__N_15inner1bE", scope: !39, file: !3, line: 47, type: !9, isLocal: true, isDefinition: true)
!39 = !DINamespace(name: "inner", scope: !36)
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = !DIGlobalVariable(name: "c", linkageName: "_ZN5outer12_GLOBAL__N_11cE", scope: !42, file: !3, line: 53, type: !9, isLocal: true, isDefinition: true)
!42 = !DINamespace(scope: !43)
!43 = !DINamespace(name: "outer", scope: null)
!44 = !{!45, !47}
!45 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !46, file:!3, line: 34)
!46 = !DIGlobalVariable(name: "global_namespace_variable_decl", linkageName: "_ZN2ns30global_namespace_variable_declE", scope: !18, file: !3, line: 28, type: !9, isLocal: false, isDefinition: false)
!47 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !43, entity: !42, file: !3, line: 43)
!48 = !{i32 2, !"Dwarf Version", i32 4}
!49 = !{i32 2, !"Debug Info Version", i32 3}
!50 = !{!"clang version 3.7.0 (trunk 234897) (llvm/trunk 234911)"}
!51 = distinct !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: !6, file: !3, line: 9, type: !11, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !10, variables: !4)
!52 = !DILocalVariable(name: "this", arg: 1, scope: !51, type: !53, flags: DIFlagArtificial | DIFlagObjectPointer)
!53 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!54 = !DIExpression()
!55 = !DILocation(line: 0, scope: !51)
!56 = !DILocation(line: 10, scope: !51)
!57 = !DILocation(line: 11, scope: !51)
!58 = distinct !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: !6, file: !3, line: 13, type: !15, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !14, variables: !4)
!59 = !DILocation(line: 14, scope: !58)
!60 = distinct !DISubprogram(name: "global_function", linkageName: "_Z15global_functionv", scope: !3, file: !3, line: 19, type: !15, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!61 = !DILocation(line: 20, scope: !60)
!62 = distinct !DISubprogram(name: "global_namespace_function", linkageName: "_ZN2ns25global_namespace_functionEv", scope: !18, file: !3, line: 24, type: !63, isLocal: false, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!63 = !DISubroutineType(types: !64)
!64 = !{null}
!65 = !DILocation(line: 25, scope: !62)
!66 = !DILocation(line: 26, scope: !62)
!67 = !DILocation(line: 42, scope: !30)
!68 = distinct !DISubprogram(name: "f7", linkageName: "_Z2f7v", scope: !3, file: !3, line: 57, type: !15, isLocal: false, isDefinition: true, scopeLine: 57, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!69 = !DILocation(line: 58, scope: !68)

