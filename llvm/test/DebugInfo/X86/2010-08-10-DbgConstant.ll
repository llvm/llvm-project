; RUN: llc  -mtriple=i686-linux -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_constant
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-f]*}}] = "ro")

define void @foo() nounwind ssp {
entry:
  call void @bar(i32 201), !dbg !8
  ret void, !dbg !8
}

declare void @bar(i32)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = metadata !{i32 786478, metadata !12, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, void ()* @foo, null, null, null, i32 3} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !12, i32 12, metadata !"clang 2.8", i1 false, metadata !"", i32 0, metadata !4, metadata !4, metadata !10, metadata !11,  metadata !11, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !12, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 786471, i32 0, metadata !1, metadata !"ro", metadata !"ro", metadata !"ro", metadata !1, i32 1, metadata !6, i1 true, i1 true, i32 201, null} ; [ DW_TAG_constant ]
!6 = metadata !{i32 786470, metadata !12, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_const_type ]
!7 = metadata !{i32 786468, metadata !12, metadata !1, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 14, metadata !9, null}
!9 = metadata !{i32 786443, metadata !12, metadata !0, i32 3, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !0}
!11 = metadata !{metadata !5}
!12 = metadata !{metadata !"/tmp/l.c", metadata !"/Volumes/Lalgate/clean/D"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
