; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@a = global i32 0, align 4
@b = global i64 0, align 8
@c = global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}

!0 = metadata !{i32 786449, metadata !22, i32 4, metadata !"clang version 3.2 (trunk 157269) (llvm/trunk 157264)", i1 false, metadata !"", i32 0, metadata !1, metadata !15, metadata !15, metadata !17,  metadata !17, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !3, metadata !8, metadata !12}
!3 = metadata !{i32 786436, metadata !4, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, metadata !5, metadata !6, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from int]
!4 = metadata !{i32 786473, metadata !22} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786472, metadata !"A1", i64 1} ; [ DW_TAG_enumerator ]
!8 = metadata !{i32 786436, metadata !4, null, metadata !"B", i32 2, i64 64, i64 64, i32 0, i32 0, metadata !9, metadata !10, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [B] [line 2, size 64, align 64, offset 0] [def] [from long unsigned int]
!9 = metadata !{i32 786468, null, null, metadata !"long unsigned int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786472, metadata !"B1", i64 1} ; [ DW_TAG_enumerator ]
!12 = metadata !{i32 786436, metadata !4, null, metadata !"C", i32 3, i64 32, i64 32, i32 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [C] [line 3, size 32, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14}
!14 = metadata !{i32 786472, metadata !"C1", i64 1} ; [ DW_TAG_enumerator ]
!15 = metadata !{i32 0}
!17 = metadata !{metadata !19, metadata !20, metadata !21}
!19 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !4, i32 4, metadata !3, i32 0, i32 1, i32* @a, null} ; [ DW_TAG_variable ]
!20 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !4, i32 5, metadata !8, i32 0, i32 1, i64* @b, null} ; [ DW_TAG_variable ]
!21 = metadata !{i32 786484, i32 0, null, metadata !"c", metadata !"c", metadata !"", metadata !4, i32 6, metadata !12, i32 0, i32 1, i32* @c, null} ; [ DW_TAG_variable ]
!22 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}

; CHECK: DW_TAG_enumeration_type [{{.*}}]
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "A")

; CHECK: DW_TAG_enumeration_type [{{.*}}] *
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]          ( .debug_str[{{.*}}] = "B")

; CHECK: DW_TAG_enumeration_type [6]
; CHECK-NOT: DW_AT_enum_class
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "C")
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
