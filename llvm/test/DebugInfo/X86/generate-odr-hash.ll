; REQUIRES: object-emission

; RUN: llc < %s -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump %t | FileCheck --check-prefix=CHECK --check-prefix=SINGLE %s
; RUN: llvm-readobj -s -t %t | FileCheck --check-prefix=OBJ_SINGLE %s

; RUN: llc < %s -split-dwarf=Enable -o %t -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu
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

%struct.bar = type { i8 }
%"class.echidna::capybara::mongoose::fluffy" = type { i32, i32 }
%"struct.<anonymous namespace>::walrus" = type { i8 }
%struct.wombat = type { %struct.anon }
%struct.anon = type { i32, i32 }
%struct.baz = type { i8 }

@b = global %struct.bar zeroinitializer, align 1
@_ZN7echidna8capybara8mongoose6animalE = global %"class.echidna::capybara::mongoose::fluffy" zeroinitializer, align 4
@w = internal global %"struct.<anonymous namespace>::walrus" zeroinitializer, align 1
@wom = global %struct.wombat zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

; Function Attrs: nounwind uwtable
define void @_Z3foov() #0 !dbg !22 {
entry:
  %b = alloca %struct.baz, align 1
  call void @llvm.dbg.declare(metadata %struct.baz* %b, metadata !46, metadata !DIExpression()), !dbg !48
  ret void, !dbg !49
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !26 {
entry:
  call void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* @w), !dbg !50
  ret void, !dbg !50
}

; Function Attrs: nounwind uwtable
define internal void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* %this) unnamed_addr #0 align 2 !dbg !27 {
entry:
  %this.addr = alloca %"struct.<anonymous namespace>::walrus"*, align 8
  store %"struct.<anonymous namespace>::walrus"* %this, %"struct.<anonymous namespace>::walrus"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.<anonymous namespace>::walrus"** %this.addr, metadata !51, metadata !DIExpression()), !dbg !53
  %this1 = load %"struct.<anonymous namespace>::walrus"*, %"struct.<anonymous namespace>::walrus"** %this.addr
  ret void, !dbg !54
}

define internal void @_GLOBAL__I_a() section ".text.startup" !dbg !36 {
entry:
  call void @__cxx_global_var_init(), !dbg !55
  ret void, !dbg !55
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, splitDebugFilename: "bar.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !38, imports: !2)
!1 = !DIFile(filename: "bar.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !6, !14, !17}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", line: 1, size: 8, align: 8, file: !5, elements: !2, identifier: "_ZTS3bar")
!5 = !DIFile(filename: "bar.h", directory: "/tmp/dbginfo")
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "fluffy", line: 13, size: 64, align: 32, file: !1, scope: !7, elements: !10, identifier: "_ZTSN7echidna8capybara8mongoose6fluffyE")
!7 = !DINamespace(name: "mongoose", line: 12, file: !1, scope: !8)
!8 = !DINamespace(name: "capybara", line: 11, file: !1, scope: !9)
!9 = !DINamespace(name: "echidna", line: 10, file: !1, scope: null)
!10 = !{!11, !13}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 14, size: 32, align: 32, flags: DIFlagPrivate, file: !1, scope: !6, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 15, size: 32, align: 32, offset: 32, flags: DIFlagPrivate, file: !1, scope: !6, baseType: !12)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "wombat", line: 31, size: 64, align: 32, file: !1, elements: !15, identifier: "_ZTS6wombat")
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a_b", line: 35, size: 64, align: 32, file: !1, scope: !14, baseType: !17)
!17 = !DICompositeType(tag: DW_TAG_structure_type, line: 32, size: 64, align: 32, file: !1, scope: !14, elements: !18, identifier: "_ZTSN6wombatUt_E")
!18 = !{!19, !20}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 33, size: 32, align: 32, file: !1, scope: !17, baseType: !12)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 34, size: 32, align: 32, offset: 32, file: !1, scope: !17, baseType: !12)
!22 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !23, type: !24, variables: !2)
!23 = !DIFile(filename: "bar.cpp", directory: "/tmp/dbginfo")
!24 = !DISubroutineType(types: !25)
!25 = !{null}
!26 = distinct !DISubprogram(name: "__cxx_global_var_init", line: 29, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 29, file: !1, scope: !23, type: !24, variables: !2)
!27 = distinct !DISubprogram(name: "walrus", linkageName: "_ZN12_GLOBAL__N_16walrusC2Ev", line: 25, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 25, file: !1, scope: !28, type: !32, declaration: !31, variables: !2)
!28 = !DICompositeType(tag: DW_TAG_structure_type, name: "walrus", line: 24, size: 8, align: 8, file: !1, scope: !29, elements: !30)
!29 = !DINamespace(line: 23, file: !1, scope: null)
!30 = !{!31}
!31 = !DISubprogram(name: "walrus", line: 25, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 25, file: !1, scope: !28, type: !32)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !28)
!36 = distinct !DISubprogram(name: "", linkageName: "_GLOBAL__I_a", line: 25, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagArtificial, isOptimized: false, unit: !0, scopeLine: 25, file: !1, scope: !23, type: !37, variables: !2)
!37 = !DISubroutineType(types: !2)
!38 = !{!39, !40, !41, !42}
!39 = !DIGlobalVariable(name: "b", line: 3, isLocal: false, isDefinition: true, scope: null, file: !23, type: !4, variable: %struct.bar* @b)
!40 = !DIGlobalVariable(name: "animal", linkageName: "_ZN7echidna8capybara8mongoose6animalE", line: 18, isLocal: false, isDefinition: true, scope: !7, file: !23, type: !6, variable: %"class.echidna::capybara::mongoose::fluffy"* @_ZN7echidna8capybara8mongoose6animalE)
!41 = !DIGlobalVariable(name: "w", line: 29, isLocal: true, isDefinition: true, scope: null, file: !23, type: !28, variable: %"struct.<anonymous namespace>::walrus"* @w)
!42 = !DIGlobalVariable(name: "wom", line: 38, isLocal: false, isDefinition: true, scope: null, file: !23, type: !14, variable: %struct.wombat* @wom)
!43 = !{i32 2, !"Dwarf Version", i32 4}
!44 = !{i32 1, !"Debug Info Version", i32 3}
!45 = !{!"clang version 3.5 "}
!46 = !DILocalVariable(name: "b", line: 7, scope: !22, file: !23, type: !47)
!47 = !DICompositeType(tag: DW_TAG_structure_type, name: "baz", line: 6, size: 8, align: 8, file: !1, scope: !22, elements: !2)
!48 = !DILocation(line: 7, scope: !22)
!49 = !DILocation(line: 8, scope: !22)
!50 = !DILocation(line: 29, scope: !26)
!51 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !27, type: !52)
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !28)
!53 = !DILocation(line: 0, scope: !27)
!54 = !DILocation(line: 25, scope: !27)
!55 = !DILocation(line: 25, scope: !36)
