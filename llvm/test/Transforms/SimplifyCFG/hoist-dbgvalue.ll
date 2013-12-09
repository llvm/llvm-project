; RUN: opt -simplifycfg -S < %s | FileCheck %s

define i32 @foo(i32 %i) nounwind ssp {
  call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !6), !dbg !7
  call void @llvm.dbg.value(metadata !8, i64 0, metadata !9), !dbg !11
  %1 = icmp ne i32 %i, 0, !dbg !12
;CHECK: call i32 (...)* @bar()
;CHECK-NEXT: llvm.dbg.value
  br i1 %1, label %2, label %4, !dbg !12

; <label>:2                                       ; preds = %0
  %3 = call i32 (...)* @bar(), !dbg !13
  call void @llvm.dbg.value(metadata !{i32 %3}, i64 0, metadata !9), !dbg !13
  br label %6, !dbg !15

; <label>:4                                       ; preds = %0
  %5 = call i32 (...)* @bar(), !dbg !16
  call void @llvm.dbg.value(metadata !{i32 %5}, i64 0, metadata !9), !dbg !16
  br label %6, !dbg !18

; <label>:6                                       ; preds = %4, %2
  %k.0 = phi i32 [ %3, %2 ], [ %5, %4 ]
  ret i32 %k.0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @bar(...)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.module.flags = !{!21}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, metadata !20, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!1 = metadata !{i32 589865, metadata !20} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !20, i32 12, metadata !"clang", i1 true, metadata !"", i32 0, metadata !8, metadata !8, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !20, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !0, metadata !"i", metadata !1, i32 16777218, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 2, i32 13, metadata !0, null}
!8 = metadata !{i32 0}
!9 = metadata !{i32 590080, metadata !10, metadata !"k", metadata !1, i32 3, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 589835, metadata !20, metadata !0, i32 2, i32 16, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 3, i32 12, metadata !10, null}
!12 = metadata !{i32 4, i32 3, metadata !10, null}
!13 = metadata !{i32 5, i32 5, metadata !14, null}
!14 = metadata !{i32 589835, metadata !20, metadata !10, i32 4, i32 10, i32 1} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 6, i32 3, metadata !14, null}
!16 = metadata !{i32 7, i32 5, metadata !17, null}
!17 = metadata !{i32 589835, metadata !20, metadata !10, i32 6, i32 10, i32 2} ; [ DW_TAG_lexical_block ]
!18 = metadata !{i32 8, i32 3, metadata !17, null}
!19 = metadata !{i32 9, i32 3, metadata !10, null}
!20 = metadata !{metadata !"b.c", metadata !"/private/tmp"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
