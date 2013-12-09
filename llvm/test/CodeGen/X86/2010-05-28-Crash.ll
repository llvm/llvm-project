; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic < %s | FileCheck %s
; Test to check separate label for inlined function argument.

define i32 @foo(i32 %y) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %y}, i64 0, metadata !0)
  %0 = tail call i32 (...)* @zoo(i32 %y) nounwind, !dbg !9 ; <i32> [#uses=1]
  ret i32 %0, !dbg !9
}

declare i32 @zoo(...)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i32 @bar(i32 %x) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !7)
  tail call void @llvm.dbg.value(metadata !11, i64 0, metadata !0) nounwind
  %0 = tail call i32 (...)* @zoo(i32 1) nounwind, !dbg !12 ; <i32> [#uses=1]
  %1 = add nsw i32 %0, %x, !dbg !13               ; <i32> [#uses=1]
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = metadata !{i32 786689, metadata !1, metadata !"y", metadata !2, i32 2, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 786478, metadata !18, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", i32 2, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32)* @foo, null, null, metadata !15, i32 2} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !18} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !18, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !19, metadata !19, metadata !17, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !18, metadata !2, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !6}
!6 = metadata !{i32 786468, metadata !18, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 786689, metadata !8, metadata !"x", metadata !2, i32 6, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 786478, metadata !18, metadata !2, metadata !"bar", metadata !"bar", metadata !"bar", i32 6, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 (i32)* @bar, null, null, metadata !16, i32 6} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 3, i32 0, metadata !10, null}
!10 = metadata !{i32 786443, metadata !18, metadata !1, i32 2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 1}
!12 = metadata !{i32 3, i32 0, metadata !10, metadata !13}
!13 = metadata !{i32 7, i32 0, metadata !14, null}
!14 = metadata !{i32 786443, metadata !18, metadata !8, i32 6, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !0}
!16 = metadata !{metadata !7}
!17 = metadata !{metadata !1, metadata !8}
!18 = metadata !{metadata !"f.c", metadata !"/tmp"}
!19 = metadata !{i32 0}

;CHECK: DEBUG_VALUE: bar:x <- E
;CHECK: Ltmp
;CHECK:	DEBUG_VALUE: foo:y <- 1{{$}}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
