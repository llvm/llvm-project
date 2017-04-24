; REQUIRES: object-emission

; RUN: llc < %s -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump %t | FileCheck --check-prefix=CHECK --check-prefix=SINGLE %s
; RUN: llvm-readobj -s -t %t | FileCheck --check-prefix=OBJ_SINGLE %s

; RUN: llc < %s -split-dwarf-file=foo.dwo -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump %t | FileCheck --check-prefix=CHECK --check-prefix=FISSION %s
; RUN: llvm-readobj -s -t %t | FileCheck --check-prefix=OBJ_FISSION %s

; Generated from bar.cpp:

; #line 1 "bar.h"
; struct bar {};
; #line 2 "bar.cpp"

; struct bar b;

; void foo(void) {
;   struct baz {};
;   baz b;
; }

; namespace echidna {
; namespace capybara {
; namespace mongoose {
; class fluffy {
;   int a;
;   int b;
; };

; fluffy animal;
; }
; }
; }

; namespace {
; struct walrus {
;   walrus() {}
; };
; }

; walrus w;

; struct wombat {
;   struct {
;     int a;
;     int b;
;   } a_b;
; };

; wombat wom;

; SINGLE-LABEL: .debug_info contents:
; FISSION-LABEL: .debug_info.dwo contents:
; CHECK: Compile Unit: length = [[CU_SIZE:[0-9a-f]+]]

; CHECK: [[BAR:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0x1d02f3be30cc5688)
; CHECK: [[FLUFFY:^0x........]]: DW_TAG_class_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0xb04af47397402e77)

; Ensure the CU-local type 'walrus' is not placed in a type unit.
; CHECK: [[WALRUS:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"walrus"
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line

; CHECK: [[WOMBAT:^0x........]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature {{.*}} (0xfd756cee88f8a118)

; SINGLE-LABEL: .debug_types contents:
; FISSION-NOT: .debug_types contents:
; FISSION-LABEL: .debug_types.dwo contents:

; Check that we generate a hash for bar and the value.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0x1d02f3be30cc5688
; CHECK: DW_TAG_structure_type
; FISSION-NEXT: DW_AT_name {{.*}} ( indexed {{.*}} "bar"
; SINGLE-NEXT: DW_AT_name {{.*}} "bar"

; Check that we generate a hash for fluffy and the value.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xb04af47397402e77
; CHECK-NOT: DW_AT_GNU_odr_signature [DW_FORM_data8]   (0x9a0124d5a0c21c52)
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"echidna"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"capybara"
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}}"mongoose"
; CHECK: DW_TAG_class_type
; CHECK-NEXT: DW_AT_name{{.*}}"fluffy"

; Check that we generate a hash for wombat and the value, but not for the
; anonymous type contained within.
; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xfd756cee88f8a118
; CHECK-NOT: DW_AT_GNU_odr_signature [DW_FORM_data8] (0x685bcc220141e9d7)
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"wombat"

; CHECK-NOT: type_signature
; CHECK-LABEL: type_signature = 0xe94f6d3843e62d6b
; CHECK: DW_TAG_type_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset] (0x00000000)
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_GNU_odr_signature
; CHECK: DW_TAG_structure_type
; The signature for the outer 'wombat' type
; CHECK: DW_AT_signature [DW_FORM_ref_sig8] (0xfd756cee88f8a118)
; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_AT_name
; CHECK-NOT: DW_AT_GNU_odr_signature
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"

; CHECK-LABEL: .debug_line contents:
; CHECK: Line table prologue
; CHECK-NOT: file_names[
; SINGLE: file_names{{.*}} bar.h
; CHECK: file_names{{.*}} bar.cpp
; CHECK-NOT: file_names[

; CHECK-LABEL: .debug_line.dwo contents:
; FISSION: Line table prologue
; FISSION: opcode_base: 1
; FISSION-NOT: standard_opcode_lengths
; FISSION-NOT: include_directories
; FISSION-NOT: file_names[
; FISSION: file_names{{.*}} bar.h
; FISSION: file_names{{.*}} bar.cpp
; FISSION-NOT: file_names[

; CHECK-LABEL: .debug_str contents:

; Use the unit size as a rough hash/identifier for the unit we're dealing with
; it happens to be unambiguous at the moment, but it's hardly ideal.
; CHECK-LABEL: .debug_pubtypes contents:
; Don't emit pubtype entries for type DIEs in the compile unit that just indirect to a type unit.
; CHECK-NEXT: unit_size = [[CU_SIZE]]
; CHECK-NEXT: Offset Name
; CHECK-DAG: [[BAR]] "bar"
; CHECK-DAG: [[WALRUS]] "(anonymous namespace)::walrus"
; CHECK-DAG: [[WOMBAT]] "wombat"
; CHECK-DAG: [[FLUFFY]] "echidna::capybara::mongoose::fluffy"

; Make sure debug_types are in comdat groups. This could be more rigid to check
; that they're the right comdat groups (each type in a separate comdat group,
; etc)
; OBJ_SINGLE: Name: .debug_types (
; OBJ_SINGLE-NOT: }
; OBJ_SINGLE: SHF_GROUP

; Fission type units don't go in comdat groups, since their linker is debug
; aware it's handled using the debug info semantics rather than raw ELF object
; semantics.
; OBJ_FISSION: Name: .debug_types.dwo (
; OBJ_FISSION-NOT: SHF_GROUP
; OBJ_FISSION: }

source_filename = "test/DebugInfo/X86/generate-odr-hash.ll"

%struct.bar = type { i8 }
%"class.echidna::capybara::mongoose::fluffy" = type { i32, i32 }
%"struct.<anonymous namespace>::walrus" = type { i8 }
%struct.wombat = type { %struct.anon }
%struct.anon = type { i32, i32 }
%struct.baz = type { i8 }

@b = global %struct.bar zeroinitializer, align 1, !dbg !0
@_ZN7echidna8capybara8mongoose6animalE = global %"class.echidna::capybara::mongoose::fluffy" zeroinitializer, align 4, !dbg !6
@w = internal global %"struct.<anonymous namespace>::walrus" zeroinitializer, align 1, !dbg !16
@wom = global %struct.wombat zeroinitializer, align 4, !dbg !25
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

; Function Attrs: nounwind uwtable
define void @_Z3foov() #0 !dbg !40 {
entry:
  %b = alloca %struct.baz, align 1
  call void @llvm.dbg.declare(metadata %struct.baz* %b, metadata !43, metadata !45), !dbg !46
  ret void, !dbg !47
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !48 {
entry:
  call void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* @w), !dbg !49
  ret void, !dbg !49
}

; Function Attrs: nounwind uwtable
define internal void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* %this) unnamed_addr #0 align 2 !dbg !50 {
entry:
  %this.addr = alloca %"struct.<anonymous namespace>::walrus"*, align 8
  store %"struct.<anonymous namespace>::walrus"* %this, %"struct.<anonymous namespace>::walrus"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.<anonymous namespace>::walrus"** %this.addr, metadata !51, metadata !45), !dbg !53
  %this1 = load %"struct.<anonymous namespace>::walrus"*, %"struct.<anonymous namespace>::walrus"** %this.addr
  ret void, !dbg !54
}

define internal void @_GLOBAL__I_a() section ".text.startup" !dbg !55 {
entry:
  call void @__cxx_global_var_init(), !dbg !57
  ret void, !dbg !57
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!34}
!llvm.module.flags = !{!37, !38}
!llvm.ident = !{!39}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "bar.cpp", directory: "/tmp/dbginfo")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !4, line: 1, size: 8, align: 8, elements: !5, identifier: "_ZTS3bar")
!4 = !DIFile(filename: "bar.h", directory: "/tmp/dbginfo")
!5 = !{}
!6 = !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "animal", linkageName: "_ZN7echidna8capybara8mongoose6animalE", scope: !8, file: !2, line: 18, type: !11, isLocal: false, isDefinition: true)
!8 = !DINamespace(name: "mongoose", scope: !9, file: !2, line: 12)
!9 = !DINamespace(name: "capybara", scope: !10, file: !2, line: 11)
!10 = !DINamespace(name: "echidna", scope: null, file: !2, line: 10)
!11 = !DICompositeType(tag: DW_TAG_class_type, name: "fluffy", scope: !8, file: !2, line: 13, size: 64, align: 32, elements: !12, identifier: "_ZTSN7echidna8capybara8mongoose6fluffyE")
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !2, line: 14, baseType: !14, size: 32, align: 32, flags: DIFlagPrivate)
!14 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !2, line: 15, baseType: !14, size: 32, align: 32, offset: 32, flags: DIFlagPrivate)
!16 = !DIGlobalVariableExpression(var: !17)
!17 = !DIGlobalVariable(name: "w", scope: null, file: !2, line: 29, type: !18, isLocal: true, isDefinition: true)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "walrus", scope: !19, file: !2, line: 24, size: 8, align: 8, elements: !20)
!19 = !DINamespace(scope: null, file: !2, line: 23)
!20 = !{!21}
!21 = !DISubprogram(name: "walrus", scope: !18, file: !2, line: 25, type: !22, isLocal: false, isDefinition: false, scopeLine: 25, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DIGlobalVariableExpression(var: !26)
!26 = !DIGlobalVariable(name: "wom", scope: null, file: !2, line: 38, type: !27, isLocal: false, isDefinition: true)
!27 = !DICompositeType(tag: DW_TAG_structure_type, name: "wombat", file: !2, line: 31, size: 64, align: 32, elements: !28, identifier: "_ZTS6wombat")
!28 = !{!29}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "a_b", scope: !27, file: !2, line: 35, baseType: !30, size: 64, align: 32)
!30 = !DICompositeType(tag: DW_TAG_structure_type, scope: !27, file: !2, line: 32, size: 64, align: 32, elements: !31, identifier: "_ZTSN6wombatUt_E")
!31 = !{!32, !33}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !30, file: !2, line: 33, baseType: !14, size: 32, align: 32)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !30, file: !2, line: 34, baseType: !14, size: 32, align: 32, offset: 32)
!34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "bar.dwo", emissionKind: FullDebug, enums: !5, retainedTypes: !35, globals: !36, imports: !5)
!35 = !{!3, !11, !27, !30}
!36 = !{!0, !6, !16, !25}
!37 = !{i32 2, !"Dwarf Version", i32 4}
!38 = !{i32 1, !"Debug Info Version", i32 3}
!39 = !{!"clang version 3.5 "}
!40 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !2, file: !2, line: 5, type: !41, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !34, variables: !5)
!41 = !DISubroutineType(types: !42)
!42 = !{null}
!43 = !DILocalVariable(name: "b", scope: !40, file: !2, line: 7, type: !44)
!44 = !DICompositeType(tag: DW_TAG_structure_type, name: "baz", scope: !40, file: !2, line: 6, size: 8, align: 8, elements: !5)
!45 = !DIExpression()
!46 = !DILocation(line: 7, scope: !40)
!47 = !DILocation(line: 8, scope: !40)
!48 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !2, file: !2, line: 29, type: !41, isLocal: true, isDefinition: true, scopeLine: 29, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !34, variables: !5)
!49 = !DILocation(line: 29, scope: !48)
!50 = distinct !DISubprogram(name: "walrus", linkageName: "_ZN12_GLOBAL__N_16walrusC2Ev", scope: !18, file: !2, line: 25, type: !22, isLocal: true, isDefinition: true, scopeLine: 25, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !34, declaration: !21, variables: !5)
!51 = !DILocalVariable(name: "this", arg: 1, scope: !50, type: !52, flags: DIFlagArtificial | DIFlagObjectPointer)
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, align: 64)
!53 = !DILocation(line: 0, scope: !50)
!54 = !DILocation(line: 25, scope: !50)
!55 = distinct !DISubprogram(linkageName: "_GLOBAL__I_a", scope: !2, file: !2, line: 25, type: !56, isLocal: true, isDefinition: true, scopeLine: 25, virtualIndex: 6, flags: DIFlagArtificial, isOptimized: false, unit: !34, variables: !5)
!56 = !DISubroutineType(types: !5)
!57 = !DILocation(line: 25, scope: !55)

