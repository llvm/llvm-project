; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

@e = global i16 0, align 2

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !"foo.cpp", metadata !"/tmp", metadata !"clang version 3.2 (trunk 165274) (llvm/trunk 165272)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ] [/tmp/foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"e", metadata !"e", metadata !"", metadata !6, i32 2, metadata !7, i32 0, i32 1, i16* @e} ; [ DW_TAG_variable ] [e] [line 2] [def]
!6 = metadata !{i32 786473, metadata !"foo.cpp", metadata !"/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786436, null, metadata !"E", metadata !6, i32 1, i64 16, i64 16, i32 0, i32 4, null, null, i32 0} ; [ DW_TAG_enumeration_type ] [E] [line 1, size 16, align 16, offset 0] [fwd] [from ]

; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_declaration
