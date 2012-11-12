; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; Radar 9511391

;CHECK:         .byte   4                       ## DW_AT_const_value
define i32 @foo() nounwind uwtable readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !8, i64 0, metadata !6), !dbg !9
  ret i32 42, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1}
!llvm.dbg.lv.foo = !{!6}

!0 = metadata !{i32 589841, i32 0, i32 12, metadata !"a.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 132191)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"", metadata !2, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 true, i32 ()* @foo, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !"a.c", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590080, metadata !7, metadata !"i", metadata !2, i32 2, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!7 = metadata !{i32 589835, metadata !1, i32 1, i32 11, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{i32 42}
!9 = metadata !{i32 2, i32 12, metadata !7, null}
!10 = metadata !{i32 3, i32 2, metadata !7, null}
