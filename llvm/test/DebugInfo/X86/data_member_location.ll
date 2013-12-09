; RUN: llc -mtriple=x86_64-linux -O0 -o - -filetype=obj < %s | llvm-dwarfdump -debug-dump=info -| FileCheck %s

; Generated from Clang with the following source:
;
; struct foo {
;   char c;
;   int i;
; };
; 
; foo f;

; CHECK: DW_AT_name {{.*}} "c"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_data_member_location {{.*}} (0x00)

; CHECK: DW_AT_name {{.*}} "i"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_data_member_location {{.*}} (0x04)

%struct.foo = type { i8, i32 }

@f = global %struct.foo zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !15}
!llvm.ident = !{!14}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !2, metadata !10, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/data_member_location.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"data_member_location.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"foo", i32 1, i64 64, i64 32, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 64, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS3foo", metadata !"c", i32 2, i64 8, i64 8, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [c] [line 2, size 8, align 8, offset 0] [from char]
!7 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!8 = metadata !{i32 786445, metadata !1, metadata !"_ZTS3foo", metadata !"i", i32 3, i64 32, i64 32, i64 32, i32 0, metadata !9} ; [ DW_TAG_member ] [i] [line 3, size 32, align 32, offset 32] [from int]
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786484, i32 0, null, metadata !"f", metadata !"f", metadata !"", metadata !12, i32 6, metadata !4, i32 0, i32 1, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!12 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/data_member_location.cpp]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{metadata !"clang version 3.4 "}

!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
