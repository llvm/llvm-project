; RUN: opt -strip-dead-debug-info -verify %s -S | FileCheck %s

; CHECK: ModuleID = '{{.*}}'
; CHECK-NOT: bar
; CHECK-NOT: abcd

@xyz = global i32 2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #0

; Function Attrs: nounwind readnone ssp
define i32 @fn() #1 {
entry:
  ret i32 0, !dbg !18
}

; Function Attrs: nounwind readonly ssp
define i32 @foo(i32 %i) #2 {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !15), !dbg !20
  %.0 = load i32* @xyz, align 4
  ret i32 %.0, !dbg !21
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone ssp }
attributes #2 = { nounwind readonly ssp }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25}

!0 = metadata !{i32 524305, metadata !1, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !23, metadata !24, null, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp//g.c] [DW_LANG_C89]
!1 = metadata !{metadata !"g.c", metadata !"/tmp/"}
!2 = metadata !{null}
!3 = metadata !{i32 524334, metadata !1, null, metadata !"bar", metadata !"bar", metadata !"", i32 5, metadata !4, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 true, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 5] [local] [def] [scope 0] [bar]
!4 = metadata !{i32 524309, metadata !1, metadata !5, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{i32 524329, metadata !1}          ; [ DW_TAG_file_type ] [/tmp//g.c]
!6 = metadata !{i32 524334, metadata !1, null, metadata !"fn", metadata !"fn", metadata !"fn", i32 6, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @fn, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [fn]
!7 = metadata !{i32 524309, metadata !1, metadata !5, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 524324, metadata !1, metadata !5, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 524334, metadata !1, null, metadata !"foo", metadata !"foo", metadata !"foo", i32 7, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32)* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 0] [foo]
!11 = metadata !{i32 524309, metadata !1, metadata !5, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !9, metadata !9}
!13 = metadata !{i32 524544, metadata !14, metadata !"bb", metadata !5, i32 5, metadata !9}
!14 = metadata !{i32 524299, metadata !1, metadata !3, i32 5, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!15 = metadata !{i32 524545, metadata !10, metadata !"i", metadata !5, i32 7, metadata !9}
!16 = metadata !{i32 524340, i32 0, metadata !5, metadata !"abcd", metadata !"abcd", metadata !"", metadata !5, i32 2, metadata !9, i1 true, i1 true, null, null}
!17 = metadata !{i32 524340, i32 0, metadata !5, metadata !"xyz", metadata !"xyz", metadata !"", metadata !5, i32 3, metadata !9, i1 false, i1 true, i32* @xyz, null}
!18 = metadata !{i32 6, i32 0, metadata !19, null}
!19 = metadata !{i32 524299, metadata !1, metadata !6, i32 6, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!20 = metadata !{i32 7, i32 0, metadata !10, null}
!21 = metadata !{i32 10, i32 0, metadata !22, null}
!22 = metadata !{i32 524299, metadata !1, metadata !10, i32 7, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!23 = metadata !{metadata !3, metadata !6, metadata !10}
!24 = metadata !{metadata !16, metadata !17}
!25 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
