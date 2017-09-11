; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu -dwarf-version=4
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s -check-prefix=PRESENT 
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s -check-prefix=ABSENT
; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-apple-darwin -dwarf-version=4
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s -check-prefix=DARWINP
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s -check-prefix=DARWINA
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

source_filename = "test/DebugInfo/X86/debug-info-static-member.ll"

%class.C = type { i32 }

@_ZN1C1aE = global i32 4, align 4, !dbg !0
@_ZN1C1bE = global i32 2, align 4, !dbg !18
@_ZN1C1cE = global i32 1, align 4, !dbg !20

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !26 {
entry:
  %retval = alloca i32, align 4
  %instance_C = alloca %class.C, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.C* %instance_C, metadata !29, metadata !30), !dbg !31
  %d = getelementptr inbounds %class.C, %class.C* %instance_C, i32 0, i32 0, !dbg !32
  store i32 8, i32* %d, align 4, !dbg !32
  %0 = load i32, i32* @_ZN1C1cE, align 4, !dbg !33
  ret i32 %0, !dbg !33
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!22}
!llvm.module.flags = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", linkageName: "_ZN1C1aE", scope: null, file: !2, line: 14, type: !3, isLocal: false, isDefinition: true, declaration: !4)
!2 = !DIFile(filename: "/usr/local/google/home/blaikie/Development/llvm/src/tools/clang/test/CodeGenCXX/debug-info-static-member.cpp", directory: "/home/blaikie/local/Development/llvm/build/clang/x86-64/Debug/llvm")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !2, line: 3, baseType: !3, flags: DIFlagPrivate | DIFlagStaticMember)
!5 = !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !2, line: 1, size: 32, align: 32, elements: !6)
!6 = !{!4, !7, !10, !11, !14, !15, !17}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "const_a", scope: !5, file: !2, line: 4, baseType: !8, flags: DIFlagPrivate | DIFlagStaticMember, extraData: i1 true)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIBasicType(name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !2, line: 6, baseType: !3, flags: DIFlagProtected | DIFlagStaticMember)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "const_b", scope: !5, file: !2, line: 7, baseType: !12, flags: DIFlagProtected | DIFlagStaticMember, extraData: float 0x40091EB860000000)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !2, line: 9, baseType: !3, flags: DIFlagPublic | DIFlagStaticMember)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "const_c", scope: !5, file: !2, line: 10, baseType: !16, flags: DIFlagPublic | DIFlagStaticMember, extraData: i32 18)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !3)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !5, file: !2, line: 11, baseType: !3, size: 32, align: 32, flags: DIFlagPublic)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = !DIGlobalVariable(name: "b", linkageName: "_ZN1C1bE", scope: null, file: !2, line: 15, type: !3, isLocal: false, isDefinition: true, declaration: !10)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = !DIGlobalVariable(name: "c", linkageName: "_ZN1C1cE", scope: null, file: !2, line: 16, type: !3, isLocal: false, isDefinition: true, declaration: !14)
!22 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.3 (trunk 171914)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !23, retainedTypes: !23, globals: !24, imports: !23)
!23 = !{}
!24 = !{!0, !18, !20}
!25 = !{i32 1, !"Debug Info Version", i32 3}
!26 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 18, type: !27, isLocal: false, isDefinition: true, scopeLine: 23, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !22, variables: !23)
!27 = !DISubroutineType(types: !28)
!28 = !{!3}
!29 = !DILocalVariable(name: "instance_C", scope: !26, file: !2, line: 20, type: !5)
!30 = !DIExpression()
!31 = !DILocation(line: 20, scope: !26)
!32 = !DILocation(line: 21, scope: !26)
!33 = !DILocation(line: 22, scope: !26)

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
