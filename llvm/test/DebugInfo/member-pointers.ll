; REQUIRES: object-emission
; XFAIL: hexagon

; RUN: llc -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]})
; CHECK: [[TYPE]]:   DW_TAG_subroutine_type
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_artificial [DW_FORM_flag
; IR generated from clang -g with the following source:
; struct S {
; };
;
; int S::*x = 0;
; void (S::*y)(int) = 0;

@x = global i64 -1, align 8
@y = global { i64, i64 } zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16}

!0 = metadata !{i32 786449, metadata !15, i32 4, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/home/blaikie/Development/scratch/simple.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !10}
!5 = metadata !{i32 786484, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !6, i32 4, metadata !7, i32 0, i32 1, i64* @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!6 = metadata !{i32 786473, metadata !15} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786463, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8, metadata !9} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from int]
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786451, metadata !15, null, metadata !"S", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !1, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [S] [line 1, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{i32 786484, i32 0, null, metadata !"y", metadata !"y", metadata !"", metadata !6, i32 5, metadata !11, i32 0, i32 1, { i64, i64 }* @y, null} ; [ DW_TAG_variable ] [y] [line 5] [def]
!11 = metadata !{i32 786463, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !12, metadata !9} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null, metadata !14, metadata !8}
!14 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from S]
!15 = metadata !{metadata !"simple.cpp", metadata !"/home/blaikie/Development/scratch"}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
