; RUN: llc -O2 -asm-verbose < %s | FileCheck %s
; Radar 8616981

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

%struct.bar = type { i32, i32 }

define i32 @foo(%struct.bar* nocapture %i) nounwind readnone optsize noinline ssp {
; CHECK: TAG_formal_parameter
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.bar* %i}, i64 0, metadata !6), !dbg !12
  ret i32 1, !dbg !13
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.foo = !{!6}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (%struct.bar*)* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"one.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"one.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 117922)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !0, metadata !"i", metadata !1, i32 3, metadata !7, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 589839, metadata !1, metadata !"", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 589843, metadata !1, metadata !"bar", metadata !1, i32 2, i64 64, i64 32, i64 0, i32 0, null, metadata !9, i32 0, null} ; [ DW_TAG_structure_type ]
!9 = metadata !{metadata !10, metadata !11}
!10 = metadata !{i32 589837, metadata !1, metadata !"x", metadata !1, i32 2, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!11 = metadata !{i32 589837, metadata !1, metadata !"y", metadata !1, i32 2, i64 32, i64 32, i64 32, i32 0, metadata !5} ; [ DW_TAG_member ]
!12 = metadata !{i32 3, i32 47, metadata !0, null}
!13 = metadata !{i32 4, i32 2, metadata !14, null}
!14 = metadata !{i32 589835, metadata !0, i32 3, i32 50, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
