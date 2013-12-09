; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=PRESENT
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=ABSENT
; RUN: llc %s -o %t -filetype=obj -O0 -mtriple=x86_64-apple-darwin
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
  call void @llvm.dbg.declare(metadata !{%class.C* %instance_C}, metadata !29), !dbg !30
  %d = getelementptr inbounds %class.C* %instance_C, i32 0, i32 0, !dbg !31
  store i32 8, i32* %d, align 4, !dbg !31
  %0 = load i32* @_ZN1C1cE, align 4, !dbg !32
  ret i32 %0, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34}

!0 = metadata !{i32 786449, metadata !33, i32 4, metadata !"clang version 3.3 (trunk 171914)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !10,  metadata !10, metadata !""} ; [ DW_TAG_compile_unit ] [/home/probinson/projects/upstream/static-member/test/debug-info-static-member.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !33, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 18, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !1, i32 23} ; [ DW_TAG_subprogram ] [line 18] [def] [scope 23] [main]
!6 = metadata !{i32 786473, metadata !33} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !12, metadata !27, metadata !28}
!12 = metadata !{i32 786484, i32 0, metadata !13, metadata !"a", metadata !"a", metadata !"_ZN1C1aE", metadata !6, i32 14, metadata !9, i32 0, i32 1, i32* @_ZN1C1aE, metadata !15} ; [ DW_TAG_variable ] [a] [line 14] [def]
!13 = metadata !{i32 786434, metadata !33, null, metadata !"C", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_class_type ] [C] [line 1, size 32, align 32, offset 0] [def] [from ]
!14 = metadata !{metadata !15, metadata !16, metadata !19, metadata !20, metadata !23, metadata !24, metadata !26}
!15 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"a", i32 3, i64 0, i64 0, i64 0, i32 4097, metadata !9, null} ; [ DW_TAG_member ] [a] [line 3, size 0, align 0, offset 0] [private] [static] [from int]
!16 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"const_a", i32 4, i64 0, i64 0, i64 0, i32 4097, metadata !17, i1 true} ; [ DW_TAG_member ] [const_a] [line 4, size 0, align 0, offset 0] [private] [static] [from ]
!17 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from bool]
!18 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!19 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"b", i32 6, i64 0, i64 0, i64 0, i32 4098, metadata !9, null} ; [ DW_TAG_member ] [b] [line 6, size 0, align 0, offset 0] [protected] [static] [from int]
!20 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"const_b", i32 7, i64 0, i64 0, i64 0, i32 4098, metadata !21, float 0x40091EB860000000} ; [ DW_TAG_member ] [const_b] [line 7, size 0, align 0, offset 0] [protected] [static] [from ]
!21 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from float]
!22 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!23 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"c", i32 9, i64 0, i64 0, i64 0, i32 4096, metadata !9, null} ; [ DW_TAG_member ] [c] [line 9, size 0, align 0, offset 0] [static] [from int]
!24 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"const_c", i32 10, i64 0, i64 0, i64 0, i32 4096, metadata !25, i32 18} ; [ DW_TAG_member ] [const_c] [line 10, size 0, align 0, offset 0] [static] [from ]
!25 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!26 = metadata !{i32 786445, metadata !33, metadata !13, metadata !"d", i32 11, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ] [d] [line 11, size 32, align 32, offset 0] [from int]
!27 = metadata !{i32 786484, i32 0, metadata !13, metadata !"b", metadata !"b", metadata !"_ZN1C1bE", metadata !6, i32 15, metadata !9, i32 0, i32 1, i32* @_ZN1C1bE, metadata !19} ; [ DW_TAG_variable ] [b] [line 15] [def]
!28 = metadata !{i32 786484, i32 0, metadata !13, metadata !"c", metadata !"c", metadata !"_ZN1C1cE", metadata !6, i32 16, metadata !9, i32 0, i32 1, i32* @_ZN1C1cE, metadata !23} ; [ DW_TAG_variable ] [c] [line 16] [def]
!29 = metadata !{i32 786688, metadata !5, metadata !"instance_C", metadata !6, i32 20, metadata !13, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [instance_C] [line 20]
!30 = metadata !{i32 20, i32 0, metadata !5, null}
!31 = metadata !{i32 21, i32 0, metadata !5, null}
!32 = metadata !{i32 22, i32 0, metadata !5, null}
!33 = metadata !{metadata !"/usr/local/google/home/blaikie/Development/llvm/src/tools/clang/test/CodeGenCXX/debug-info-static-member.cpp", metadata !"/home/blaikie/local/Development/llvm/build/clang/x86-64/Debug/llvm"}
; PRESENT verifies that static member declarations have these attributes:
; external, declaration, accessibility, and either DW_AT_MIPS_linkage_name
; (for variables) or DW_AT_const_value (for constants).
;
; PRESENT:      .debug_info contents:
; PRESENT:      DW_TAG_class_type
; PRESENT-NEXT: DW_AT_name {{.*}} "C"
; PRESENT:      0x[[DECL_A:[0-9a-f]+]]: DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "a"
; PRESENT:      DW_AT_external
; PRESENT:      DW_AT_declaration
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x03)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_a"
; PRESENT:      DW_AT_external
; PRESENT:      DW_AT_declaration
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x03)
; PRESENT:      DW_AT_const_value {{.*}} (1)
; PRESENT:      0x[[DECL_B:[0-9a-f]+]]: DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "b"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x02)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_b"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x02)
; PRESENT:      DW_AT_const_value {{.*}} (0x4048f5c3)
; PRESENT:      0x[[DECL_C:[0-9a-f]+]]: DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "c"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "const_c"
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; PRESENT:      DW_AT_const_value {{.*}} (18)
; While we're here, a normal member has data_member_location and
; accessibility attributes.
; PRESENT:      DW_TAG_member
; PRESENT-NEXT: DW_AT_name {{.*}} "d"
; PRESENT:      DW_AT_data_member_location
; PRESENT:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; PRESENT:      NULL
; Definitions point back to their declarations, and have a location.
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} {0x[[DECL_A]]}
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1aE"
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} {0x[[DECL_B]]}
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1bE"
; PRESENT:      DW_TAG_variable
; PRESENT-NEXT: DW_AT_specification {{.*}} {0x[[DECL_C]]}
; PRESENT-NEXT: DW_AT_location
; PRESENT-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1cE"

; For Darwin gdb:
; DARWINP:      .debug_info contents:
; DARWINP:      DW_TAG_class_type
; DARWINP-NEXT: DW_AT_name {{.*}} "C"
; DARWINP:      0x[[DECL_A:[0-9a-f]+]]: DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "a"
; DARWINP:      DW_AT_external
; DARWINP:      DW_AT_declaration
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x03)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_a"
; DARWINP:      DW_AT_external
; DARWINP:      DW_AT_declaration
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x03)
; DARWINP:      DW_AT_const_value {{.*}} (1)
; DARWINP:      0x[[DECL_B:[0-9a-f]+]]: DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "b"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x02)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_b"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x02)
; DARWINP:      DW_AT_const_value {{.*}} (0x4048f5c3)
; DARWINP:      0x[[DECL_C:[0-9a-f]+]]: DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "c"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "const_c"
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; DARWINP:      DW_AT_const_value {{.*}} (18)
; While we're here, a normal member has data_member_location and
; accessibility attributes.
; DARWINP:      DW_TAG_member
; DARWINP-NEXT: DW_AT_name {{.*}} "d"
; DARWINP:      DW_AT_data_member_location
; DARWINP:      DW_AT_accessibility [DW_FORM_data1]   (0x01)
; DARWINP:      NULL
; Definitions point back to their declarations, and have a location.
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} {0x[[DECL_A]]}
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1aE"
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} {0x[[DECL_B]]}
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1bE"
; DARWINP:      DW_TAG_variable
; DARWINP-NEXT: DW_AT_specification {{.*}} {0x[[DECL_C]]}
; DARWINP-NEXT: DW_AT_location
; DARWINP-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1C1cE"

; ABSENT verifies that static member declarations do not have either
; DW_AT_location or DW_AT_data_member_location; also, variables do not
; have DW_AT_const_value and constants do not have DW_AT_MIPS_linkage_name.
;
; ABSENT:      .debug_info contents:
; ABSENT:      DW_TAG_member
; ABSENT:      DW_AT_name {{.*}} "a"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_a"
; ABSENT-NOT:  DW_AT_MIPS_linkage_name
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "b"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_b"
; ABSENT-NOT:  DW_AT_MIPS_linkage_name
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "c"
; ABSENT-NOT:  DW_AT_const_value
; ABSENT-NOT:  location
; ABSENT:      DW_AT_name {{.*}} "const_c"
; ABSENT-NOT:  DW_AT_MIPS_linkage_name
; ABSENT-NOT:  location
; While we're here, a normal member does not have a linkage name, constant
; value, or DW_AT_location.
; ABSENT:      DW_AT_name {{.*}} "d"
; ABSENT-NOT:  DW_AT_MIPS_linkage_name
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
; DARWINA-NOT:  DW_AT_MIPS_linkage_name
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "b"
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "const_b"
; DARWINA-NOT:  DW_AT_MIPS_linkage_name
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "c"
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  location
; DARWINA:      DW_AT_name {{.*}} "const_c"
; DARWINA-NOT:  DW_AT_MIPS_linkage_name
; DARWINA-NOT:  location
; While we're here, a normal member does not have a linkage name, constant
; value, or DW_AT_location.
; DARWINA:      DW_AT_name {{.*}} "d"
; DARWINA-NOT:  DW_AT_MIPS_linkage_name
; DARWINA-NOT:  DW_AT_const_value
; DARWINA-NOT:  DW_AT_location
; DARWINA:      NULL
!34 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
