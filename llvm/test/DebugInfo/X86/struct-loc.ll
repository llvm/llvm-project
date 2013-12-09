; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Make sure that structures have a decl file and decl line attached.
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_decl_file
; CHECK: DW_AT_decl_line
; CHECK: DW_TAG_member

%struct.foo = type { i32 }

@f = common global %struct.foo zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = metadata !{i32 786449, metadata !11, i32 12, metadata !"clang version 3.1 (trunk 152837) (llvm/trunk 152845)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"f", metadata !"f", metadata !"", metadata !6, i32 5, metadata !7, i32 0, i32 1, %struct.foo* @f, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 786473, metadata !11} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786451, metadata !11, null, metadata !"foo", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786445, metadata !11, metadata !7, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !"struct_bug.c", metadata !"/Users/echristo/tmp"}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
