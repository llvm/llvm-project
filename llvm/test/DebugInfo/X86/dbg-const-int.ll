; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; Radar 9511391

; CHECK: DW_TAG_variable
; CHECK: "i"
; CHECK: DW_AT_const_value [DW_FORM_sdata]   (42)

define i32 @foo() nounwind uwtable readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !8, i64 0, metadata !6), !dbg !9
  ret i32 42, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = metadata !{i32 786449, metadata !13, i32 12, metadata !"clang version 3.0 (trunk 132191)", i1 true, metadata !"", i32 0, metadata !14, metadata !14, metadata !11, null,  null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 786478, metadata !13, metadata !2, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, i32 ()* @foo, null, null, metadata !12, i32 0} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!2 = metadata !{i32 786473, metadata !13} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786453, metadata !13, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786688, metadata !7, metadata !"i", metadata !2, i32 2, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!7 = metadata !{i32 786443, metadata !13, metadata !1, i32 1, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 42}
!9 = metadata !{i32 2, i32 12, metadata !7, null}
!10 = metadata !{i32 3, i32 2, metadata !7, null}
!11 = metadata !{metadata !1}
!12 = metadata !{metadata !6}
!13 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!14 = metadata !{i32 0}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
