; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=PRESENT 
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=ABSENT
; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-apple-darwin -dwarf-version=4
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=DARWINP
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=DARWINA
; Verify that attributes we do want are PRESENT;
; verify that attributes we don't want are ABSENT.
; It's a lot easier to do this in two passes than in one.
; PR14471

; LLVM IR generated using: clang -emit-llvm -S -g
; (with the Clang part of this patch applied).
;
; class C
; {
;   static int a;
;   const static bool const_a = true;
; protected:
;   static int b;
;   const static float const_b = 3.14;
; public:
;   static int c;
;   const static int const_c = 18;
;   int d;
; };
; 
; int C::a = 4;
; int C::b = 2;
; int C::c = 1;
; 
; int main()
; {
;         C instance_C;
;         instance_C.d = 8;
;         return C::c;
; }

%class.C = type { i32 }

@_ZN1C1aE = global i32 4, align 4, !dbg !12
@_ZN1C1bE = global i32 2, align 4, !dbg !27
@_ZN1C1cE = global i32 1, align 4, !dbg !28

define i32 @main() nounwind uwtable !dbg !5 {
entry:
  %retval = alloca i32, align 4
  %instance_C = alloca %class.C, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.C* %instance_C, metadata !29, metadata !DIExpression()), !dbg !30
  %d = getelementptr inbounds %class.C, %class.C* %instance_C, i32 0, i32 0, !dbg !31
  store i32 8, i32* %d, align 4, !dbg !31
  %0 = load i32, i32* @_ZN1C1cE, align 4, !dbg !32
  ret i32 %0, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 171914)", isOptimized: false, emissionKind: FullDebug, file: !33, enums: !1, retainedTypes: !1, globals: !10, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "main", line: 18, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 23, file: !33, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "/usr/local/google/home/blaikie/Development/llvm/src/tools/clang/test/CodeGenCXX/debug-info-static-member.cpp", directory: "/home/blaikie/local/Development/llvm/build/clang/x86-64/Debug/llvm")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!12, !27, !28}
!12 = !DIGlobalVariable(name: "a", linkageName: "_ZN1C1aE", line: 14, isLocal: false, isDefinition: true, scope: null, file: !6, type: !9, declaration: !15)
!13 = !DICompositeType(tag: DW_TAG_class_type, name: "C", line: 1, size: 32, align: 32, file: !33, elements: !14)
!14 = !{!15, !16, !19, !20, !23, !24, !26}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 3, flags: DIFlagPrivate | DIFlagStaticMember, file: !33, scope: !13, baseType: !9)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "const_a", line: 4, flags: DIFlagPrivate | DIFlagStaticMember, file: !33, scope: !13, baseType: !17, extraData: i1 true)
!17 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !18)
!18 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 6, flags: DIFlagProtected | DIFlagStaticMember, file: !33, scope: !13, baseType: !9)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "const_b", line: 7, flags: DIFlagProtected | DIFlagStaticMember, file: !33, scope: !13, baseType: !21, extraData: float 0x40091EB860000000)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!22 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "c", line: 9, flags: DIFlagPublic | DIFlagStaticMember, file: !33, scope: !13, baseType: !9)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "const_c", line: 10, flags: DIFlagPublic | DIFlagStaticMember, file: !33, scope: !13, baseType: !25, extraData: i32 18)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "d", line: 11, size: 32, align: 32, flags: DIFlagPublic, file: !33, scope: !13, baseType: !9)
!27 = !DIGlobalVariable(name: "b", linkageName: "_ZN1C1bE", line: 15, isLocal: false, isDefinition: true, scope: null, file: !6, type: !9, declaration: !19)
!28 = !DIGlobalVariable(name: "c", linkageName: "_ZN1C1cE", line: 16, isLocal: false, isDefinition: true, scope: null, file: !6, type: !9, declaration: !23)
!29 = !DILocalVariable(name: "instance_C", line: 20, scope: !5, file: !6, type: !13)
!30 = !DILocation(line: 20, scope: !5)
!31 = !DILocation(line: 21, scope: !5)
!32 = !DILocation(line: 22, scope: !5)
!33 = !DIFile(filename: "/usr/local/google/home/blaikie/Development/llvm/src/tools/clang/test/CodeGenCXX/debug-info-static-member.cpp", directory: "/home/blaikie/local/Development/llvm/build/clang/x86-64/Debug/llvm")
; PRESENT verifies that static member declarations have these attributes:
; external, declaration, accessibility, and either DW_AT_linkage_name
; (for variables) or DW_AT_const_value (for constants).
;
; PRESENT:      .debug_info contents:
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} "a"
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1aE"
; PRESENT:      DW_TAG_class_type
; PRESENT-NEXT: DW_AT_name {{.*}} "C"
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "a"
; PRESENT:      DW_AT_external
; PRESENT:      DW_AT_declaration
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_private)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_a"
; PRESENT:      DW_AT_external
; PRESENT:      DW_AT_declaration
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_private)
; PRESENT:      DW_AT_const_value {{.*}} (1)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "b"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_protected)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_b"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_protected)
; PRESENT:      DW_AT_const_value [DW_FORM_udata] (1078523331)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "c"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_c"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; PRESENT:      DW_AT_const_value {{.*}} (18)
; While we're here, a normal member has data_member_location and
; accessibility attributes.
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "d"
; PRESENT:      DW_AT_data_member_location
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; PRESENT:      NULL
; Definitions point back to their declarations, and have a location.
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} "b"
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1bE"
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} "c"
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1cE"

; For Darwin gdb:
; DARWINP:      .debug_info contents:
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} "a"
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1aE"
; DARWINP:      DW_TAG_class_type
; DARWINP-NEXT: DW_AT_name {{.*}} "C"
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "a"
; DARWINP:      DW_AT_external
; DARWINP:      DW_AT_declaration
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_private)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_a"
; DARWINP:      DW_AT_external
; DARWINP:      DW_AT_declaration
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_private)
; DARWINP:      DW_AT_const_value {{.*}} (1)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "b"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_protected)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_b"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_protected)
; DARWINP:      DW_AT_const_value [DW_FORM_udata] (1078523331)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "c"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_c"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; DARWINP:      DW_AT_const_value {{.*}} (18)
; While we're here, a normal member has data_member_location and
; accessibility attributes.
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "d"
; DARWINP:      DW_AT_data_member_location
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; DARWINP:      NULL
; Definitions point back to their declarations, and have a location.
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} "b"
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1bE"
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} "c"
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_linkage_name {{.*}} "_ZN1C1cE"

; ABSENT verifies that static member declarations do not have either
; DW_AT_location or DW_AT_data_member_location; also, variables do not
; have DW_AT_const_value and constants do not have DW_AT_linkage_name.
;
; ABSENT:      .debug_info contents:
; ABSENT:      DW_TAG_member
; ABSENT:      DW_AT_name {{.*}} "a"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_a"
; ABSENT-NOT:  DW_AT_linkage_name
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "b"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_b"
; ABSENT-NOT:  DW_AT_linkage_name
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "c"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_c"
; ABSENT-NOT:  DW_AT_linkage_name
; ABSENT-NOT:  location
; While we're here, a normal member does not have a linkage name, constant
; value, or DW_AT_location.
; ABSENT:      DW_AT_name {{.*}} "d"
; ABSENT-NOT:  DW_AT_linkage_name
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  DW_AT_location
; ABSENT:      NULL

; For Darwin gdb:
; DARWINA:      .debug_info contents:
; DARWINA:      DW_TAG_member
; DARWINA:      DW_AT_name {{.*}} "a"
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "const_a"
; DARWINA-NOT:  DW_AT_linkage_name
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "b"
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "const_b"
; DARWINA-NOT:  DW_AT_linkage_name
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "c"
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "const_c"
; DARWINA-NOT:  DW_AT_linkage_name
; DARWINA-NOT:  location
; While we're here, a normal member does not have a linkage name, constant
; value, or DW_AT_location.
; DARWINA:      DW_AT_name {{.*}} "d"
; DARWINA-NOT:  DW_AT_linkage_name
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  DW_AT_location
; DARWINA:      NULL
!34 = !{i32 1, !"Debug Info Version", i32 3}
