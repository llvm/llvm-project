; REQUIRES: object-emission

; RUN: llc -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]})
; CHECK: [[TYPE]]:   DW_TAG_base_type
; IR generated from clang -g with the following source:
; struct Foo {
;   int e;
; };
; int Foo:*x = 0;

@x = global i64 -1, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !2, metadata !5, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cpp", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"Foo", i32 1, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, metadata !"_ZTS3Foo"} ; [ DW_TAG_structure_type ] [Foo] [line 1, size 0, align 0, offset 0] [decl] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786484, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !7, i32 4, metadata !8, i32 0, i32 1, i64* @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!7 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [foo.cpp]
!8 = metadata !{i32 786463, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9, metadata !"_ZTS3Foo"} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from int]
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
