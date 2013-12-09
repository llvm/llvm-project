; RUN: llc < %s -break-anti-dependencies=all -march=ppc64 -mcpu=g5 | FileCheck %s
; CHECK-LABEL: main:

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !15), !dbg !17
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !16), !dbg !18
  %add = add nsw i32 %argc, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = metadata !{i32 720913, metadata !21, i32 12, metadata !"clang version 3.1", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1, metadata !"", metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, metadata !21, null, metadata !"main", metadata !"main", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i8**)* @main, null, null, metadata !13, i32 0} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !21} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9, metadata !10}
!9 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 720911, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 720911, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 720932, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 8} ; [ DW_TAG_base_type ]
!13 = metadata !{metadata !14}
!14 = metadata !{metadata !15, metadata !16}
!15 = metadata !{i32 721153, metadata !5, metadata !"argc", metadata !6, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!16 = metadata !{i32 721153, metadata !5, metadata !"argv", metadata !6, i32 33554433, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 1, i32 14, metadata !5, null}
!18 = metadata !{i32 1, i32 26, metadata !5, null}
!19 = metadata !{i32 2, i32 3, metadata !20, null}
!20 = metadata !{i32 720907, metadata !21, metadata !5, i32 1, i32 34, i32 0} ; [ DW_TAG_lexical_block ]
!21 = metadata !{metadata !"dbg.c", metadata !"/src"}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
