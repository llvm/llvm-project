; RUN: llc -O0 < %s | FileCheck %s
; Radar 10464995
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@s = common global [4294967296 x i8] zeroinitializer, align 16
;CHECK: .long	4294967295

define void @bar() nounwind uwtable ssp {
entry:
  store i8 97, i8* getelementptr inbounds ([4294967296 x i8]* @s, i32 0, i64 0), align 1, !dbg !18
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"small.c", metadata !"/private/tmp", metadata !"clang version 3.1 (trunk 144833)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !11} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"bar", metadata !"bar", metadata !"", metadata !6, i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void ()* @bar, null, null, metadata !9} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"small.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{metadata !10}
!10 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !13}
!13 = metadata !{i32 720948, i32 0, null, metadata !"s", metadata !"s", metadata !"", metadata !6, i32 2, metadata !14, i32 0, i32 1, [4294967296 x i8]* @s} ; [ DW_TAG_variable ]
!14 = metadata !{i32 720897, null, metadata !"", null, i32 0, i64 34359738368, i64 8, i32 0, i32 0, metadata !15, metadata !16, i32 0, i32 0} ; [ DW_TAG_array_type ]
!15 = metadata !{i32 720932, null, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 720929, i64 0, i64 4294967295} ; [ DW_TAG_subrange_type ]
!18 = metadata !{i32 5, i32 3, metadata !19, null}
!19 = metadata !{i32 720907, metadata !5, i32 4, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!20 = metadata !{i32 6, i32 1, metadata !19, null}
