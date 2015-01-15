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
define void @_Z3foov() #0 {
entry:
  %b = alloca %struct.baz, align 1
  call void @llvm.dbg.declare(metadata %struct.baz* %b, metadata !46, metadata !{!"0x102"}), !dbg !48
  ret void, !dbg !49
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* @w), !dbg !50
  ret void, !dbg !50
}

; Function Attrs: nounwind uwtable
define internal void @_ZN12_GLOBAL__N_16walrusC2Ev(%"struct.<anonymous namespace>::walrus"* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %"struct.<anonymous namespace>::walrus"*, align 8
  store %"struct.<anonymous namespace>::walrus"* %this, %"struct.<anonymous namespace>::walrus"** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %"struct.<anonymous namespace>::walrus"** %this.addr, metadata !51, metadata !{!"0x102"}), !dbg !53
  %this1 = load %"struct.<anonymous namespace>::walrus"** %this.addr
  ret void, !dbg !54
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !55
  ret void, !dbg !55
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = !{!"0x11\004\00clang version 3.5 \000\00\000\00bar.dwo\000", !1, !2, !3, !21, !38, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/bar.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"bar.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !6, !14, !17}
!4 = !{!"0x13\00bar\001\008\008\000\000\000", !5, null, null, !2, null, null, !"_ZTS3bar"} ; [ DW_TAG_structure_type ] [bar] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!"bar.h", !"/tmp/dbginfo"}
!6 = !{!"0x2\00fluffy\0013\0064\0032\000\000\000", !1, !7, null, !10, null, null, !"_ZTSN7echidna8capybara8mongoose6fluffyE"} ; [ DW_TAG_class_type ] [fluffy] [line 13, size 64, align 32, offset 0] [def] [from ]
!7 = !{!"0x39\00mongoose\0012", !1, !8} ; [ DW_TAG_namespace ] [mongoose] [line 12]
!8 = !{!"0x39\00capybara\0011", !1, !9} ; [ DW_TAG_namespace ] [capybara] [line 11]
!9 = !{!"0x39\00echidna\0010", !1, null} ; [ DW_TAG_namespace ] [echidna] [line 10]
!10 = !{!11, !13}
!11 = !{!"0xd\00a\0014\0032\0032\000\001", !1, !"_ZTSN7echidna8capybara8mongoose6fluffyE", !12} ; [ DW_TAG_member ] [a] [line 14, size 32, align 32, offset 0] [private] [from int]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{!"0xd\00b\0015\0032\0032\0032\001", !1, !"_ZTSN7echidna8capybara8mongoose6fluffyE", !12} ; [ DW_TAG_member ] [b] [line 15, size 32, align 32, offset 32] [private] [from int]
!14 = !{!"0x13\00wombat\0031\0064\0032\000\000\000", !1, null, null, !15, null, null, !"_ZTS6wombat"} ; [ DW_TAG_structure_type ] [wombat] [line 31, size 64, align 32, offset 0] [def] [from ]
!15 = !{!16}
!16 = !{!"0xd\00a_b\0035\0064\0032\000\000", !1, !"_ZTS6wombat", !"_ZTSN6wombatUt_E"} ; [ DW_TAG_member ] [a_b] [line 35, size 64, align 32, offset 0] [from _ZTSN6wombatUt_E]
!17 = !{!"0x13\00\0032\0064\0032\000\000\000", !1, !"_ZTS6wombat", null, !18, null, null, !"_ZTSN6wombatUt_E"} ; [ DW_TAG_structure_type ] [line 32, size 64, align 32, offset 0] [def] [from ]
!18 = !{!19, !20}
!19 = !{!"0xd\00a\0033\0032\0032\000\000", !1, !"_ZTSN6wombatUt_E", !12} ; [ DW_TAG_member ] [a] [line 33, size 32, align 32, offset 0] [from int]
!20 = !{!"0xd\00b\0034\0032\0032\0032\000", !1, !"_ZTSN6wombatUt_E", !12} ; [ DW_TAG_member ] [b] [line 34, size 32, align 32, offset 32] [from int]
!21 = !{!22, !26, !27, !36}
!22 = !{!"0x2e\00foo\00foo\00_Z3foov\005\000\001\000\006\00256\000\005", !1, !23, !24, null, void ()* @_Z3foov, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!23 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/bar.cpp]
!24 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = !{null}
!26 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\0029\001\001\000\006\00256\000\0029", !1, !23, !24, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 29] [local] [def] [__cxx_global_var_init]
!27 = !{!"0x2e\00walrus\00walrus\00_ZN12_GLOBAL__N_16walrusC2Ev\0025\001\001\000\006\00256\000\0025", !1, !28, !32, null, void (%"struct.<anonymous namespace>::walrus"*)* @_ZN12_GLOBAL__N_16walrusC2Ev, null, !31, !2} ; [ DW_TAG_subprogram ] [line 25] [local] [def] [walrus]
!28 = !{!"0x13\00walrus\0024\008\008\000\000\000", !1, !29, null, !30, null, null, null} ; [ DW_TAG_structure_type ] [walrus] [line 24, size 8, align 8, offset 0] [def] [from ]
!29 = !{!"0x39\00\0023", !1, null} ; [ DW_TAG_namespace ] [line 23]
!30 = !{!31}
!31 = !{!"0x2e\00walrus\00walrus\00\0025\000\000\000\006\00256\000\0025", !1, !28, !32, null, null, null, i32 0, !35} ; [ DW_TAG_subprogram ] [line 25] [walrus]
!32 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = !{null, !34}
!34 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !28} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from walrus]
!35 = !{i32 786468}
!36 = !{!"0x2e\00\00\00_GLOBAL__I_a\0025\001\001\000\006\0064\000\0025", !1, !23, !37, null, void ()* @_GLOBAL__I_a, null, null, !2} ; [ DW_TAG_subprogram ] [line 25] [local] [def]
!37 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!38 = !{!39, !40, !41, !42}
!39 = !{!"0x34\00b\00b\00\003\000\001", null, !23, !4, %struct.bar* @b, null} ; [ DW_TAG_variable ] [b] [line 3] [def]
!40 = !{!"0x34\00animal\00animal\00_ZN7echidna8capybara8mongoose6animalE\0018\000\001", !7, !23, !6, %"class.echidna::capybara::mongoose::fluffy"* @_ZN7echidna8capybara8mongoose6animalE, null} ; [ DW_TAG_variable ] [animal] [line 18] [def]
!41 = !{!"0x34\00w\00w\00\0029\001\001", null, !23, !28, %"struct.<anonymous namespace>::walrus"* @w, null} ; [ DW_TAG_variable ] [w] [line 29] [local] [def]
!42 = !{!"0x34\00wom\00wom\00\0038\000\001", null, !23, !14, %struct.wombat* @wom, null} ; [ DW_TAG_variable ] [wom] [line 38] [def]
!43 = !{i32 2, !"Dwarf Version", i32 4}
!44 = !{i32 1, !"Debug Info Version", i32 2}
!45 = !{!"clang version 3.5 "}
!46 = !{!"0x100\00b\007\000", !22, !23, !47} ; [ DW_TAG_auto_variable ] [b] [line 7]
!47 = !{!"0x13\00baz\006\008\008\000\000\000", !1, !22, null, !2, null, null, null} ; [ DW_TAG_structure_type ] [baz] [line 6, size 8, align 8, offset 0] [def] [from ]
!48 = !MDLocation(line: 7, scope: !22)
!49 = !MDLocation(line: 8, scope: !22)
!50 = !MDLocation(line: 29, scope: !26)
!51 = !{!"0x101\00this\0016777216\001088", !27, null, !52} ; [ DW_TAG_arg_variable ] [this] [line 0]
!52 = !{!"0xf\00\000\0064\0064\000\000", null, null, !28} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from walrus]
!53 = !MDLocation(line: 0, scope: !27)
!54 = !MDLocation(line: 25, scope: !27)
!55 = !MDLocation(line: 25, scope: !36)
