; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata !0, i64 0, metadata !1), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!16}

!0 = metadata !{i128 42 }
!1 = metadata !{i32 786688, metadata !2, metadata !"MAX", metadata !4, i32 29, metadata !8, i32 0, null} ; [ DW_TAG_auto_variable ]
!2 = metadata !{i32 786443, metadata !13, metadata !3, i32 26, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 786478, metadata !13, metadata !4, metadata !"__foo", metadata !"__foo", metadata !"__foo", i32 26, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false, i32 0, i128 (i128, i128)* @__foo, null, null, null, i32 26} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 786473, metadata !13} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 786449, metadata !13, i32 1, metadata !"clang", i1 true, metadata !"", i32 0, metadata !15, metadata !15, metadata !12, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 786453, metadata !13, metadata !4, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8, metadata !8}
!8 = metadata !{i32 786454, metadata !14, metadata !4, metadata !"ti_int", i32 78, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!9 = metadata !{i32 786473, metadata !14} ; [ DW_TAG_file_type ]
!10 = metadata !{i32 786468, metadata !13, metadata !4, metadata !"", i32 0, i64 128, i64 128, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 29, i32 0, metadata !2, null}
!12 = metadata !{metadata !3}
!13 = metadata !{metadata !"foo.c", metadata !"/tmp"}
!14 = metadata !{metadata !"myint.h", metadata !"/tmp"}
!15 = metadata !{i32 0}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
