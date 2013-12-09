; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@e = global i16 0, align 2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = metadata !{i32 786449, metadata !8, i32 4, metadata !"clang version 3.2 (trunk 165274) (llvm/trunk 165272)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"e", metadata !"e", metadata !"", metadata !6, i32 2, metadata !7, i32 0, i32 1, i16* @e, null} ; [ DW_TAG_variable ] [e] [line 2] [def]
!6 = metadata !{i32 786473, metadata !8} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786436, metadata !8, null, metadata !"E", i32 1, i64 16, i64 16, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [E] [line 1, size 16, align 16, offset 0] [decl] [from ]
!8 = metadata !{metadata !"foo.cpp", metadata !"/tmp"}

; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_declaration
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
