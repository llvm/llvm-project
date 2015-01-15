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

@_ZN1C1aE = global i32 4, align 4
@_ZN1C1bE = global i32 2, align 4
@_ZN1C1cE = global i32 1, align 4

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %instance_C = alloca %class.C, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata %class.C* %instance_C, metadata !29, metadata !{!"0x102"}), !dbg !30
  %d = getelementptr inbounds %class.C* %instance_C, i32 0, i32 0, !dbg !31
  store i32 8, i32* %d, align 4, !dbg !31
  %0 = load i32* @_ZN1C1cE, align 4, !dbg !32
  ret i32 %0, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 171914)\000\00\000\00\000", !33, !1, !1, !3, !10,  !1} ; [ DW_TAG_compile_unit ] [/home/probinson/projects/upstream/static-member/test/debug-info-static-member.cpp] [DW_LANG_C_plus_plus]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00main\00main\00\0018\000\001\000\006\00256\000\0023", !33, !6, !7, null, i32 ()* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 18] [def] [scope 23] [main]
!6 = !{!"0x29", !33} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!12, !27, !28}
!12 = !{!"0x34\00a\00a\00_ZN1C1aE\0014\000\001", null, !6, !9, i32* @_ZN1C1aE, !15} ; [ DW_TAG_variable ] [a] [line 14] [def]
!13 = !{!"0x2\00C\001\0032\0032\000\000\000", !33, null, null, !14, null, null, null} ; [ DW_TAG_class_type ] [C] [line 1, size 32, align 32, offset 0] [def] [from ]
!14 = !{!15, !16, !19, !20, !23, !24, !26}
!15 = !{!"0xd\00a\003\000\000\000\004097", !33, !13, !9, null} ; [ DW_TAG_member ] [a] [line 3, size 0, align 0, offset 0] [private] [static] [from int]
!16 = !{!"0xd\00const_a\004\000\000\000\004097", !33, !13, !17, i1 true} ; [ DW_TAG_member ] [const_a] [line 4, size 0, align 0, offset 0] [private] [static] [from ]
!17 = !{!"0x26\00\000\000\000\000\000", null, null, !18} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from bool]
!18 = !{!"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!19 = !{!"0xd\00b\006\000\000\000\004098", !33, !13, !9, null} ; [ DW_TAG_member ] [b] [line 6, size 0, align 0, offset 0] [protected] [static] [from int]
!20 = !{!"0xd\00const_b\007\000\000\000\004098", !33, !13, !21, float 0x40091EB860000000} ; [ DW_TAG_member ] [const_b] [line 7, size 0, align 0, offset 0] [protected] [static] [from ]
!21 = !{!"0x26\00\000\000\000\000\000", null, null, !22} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from float]
!22 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!23 = !{!"0xd\00c\009\000\000\000\004099", !33, !13, !9, null} ; [ DW_TAG_member ] [c] [line 9, size 0, align 0, offset 0] [static] [from int]
!24 = !{!"0xd\00const_c\0010\000\000\000\004099", !33, !13, !25, i32 18} ; [ DW_TAG_member ] [const_c] [line 10, size 0, align 0, offset 0] [static] [from ]
!25 = !{!"0x26\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!26 = !{!"0xd\00d\0011\0032\0032\000\003", !33, !13, !9} ; [ DW_TAG_member ] [d] [line 11, size 32, align 32, offset 0] [from int]
!27 = !{!"0x34\00b\00b\00_ZN1C1bE\0015\000\001", null, !6, !9, i32* @_ZN1C1bE, !19} ; [ DW_TAG_variable ] [b] [line 15] [def]
!28 = !{!"0x34\00c\00c\00_ZN1C1cE\0016\000\001", null, !6, !9, i32* @_ZN1C1cE, !23} ; [ DW_TAG_variable ] [c] [line 16] [def]
!29 = !{!"0x100\00instance_C\0020\000", !5, !6, !13} ; [ DW_TAG_auto_variable ] [instance_C] [line 20]
!30 = !MDLocation(line: 20, scope: !5)
!31 = !MDLocation(line: 21, scope: !5)
!32 = !MDLocation(line: 22, scope: !5)
!33 = !{!"/usr/local/google/home/blaikie/Development/llvm/src/tools/clang/test/CodeGenCXX/debug-info-static-member.cpp", !"/home/blaikie/local/Development/llvm/build/clang/x86-64/Debug/llvm"}
; PRESENT verifies that static member declarations have these attributes:
; external, declaration, accessibility, and either DW_AT_MIPS_linkage_name
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
!34 = !{i32 1, !"Debug Info Version", i32 2}
