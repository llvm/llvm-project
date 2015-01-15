; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -regalloc=basic < %s | FileCheck %s
; Test to check separate label for inlined function argument.

define i32 @foo(i32 %y) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !0, metadata !{!"0x102"})
  %0 = tail call i32 (...)* @zoo(i32 %y) nounwind, !dbg !9 ; <i32> [#uses=1]
  ret i32 %0, !dbg !9
}

declare i32 @zoo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar(i32 %x) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !7, metadata !{!"0x102"})
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !0, metadata !{!"0x102"}) nounwind
  %0 = tail call i32 (...)* @zoo(i32 1) nounwind, !dbg !12 ; <i32> [#uses=1]
  %1 = add nsw i32 %0, %x, !dbg !13               ; <i32> [#uses=1]
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = !{!"0x101\00y\002\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\001\002", !18, !2, !4, null, i32 (i32)* @foo, null, null, !15} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\001", !18, !19, !19, !17, null,  null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !18, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !6}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", !18, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0x101\00x\006\000", !8, !2, !6} ; [ DW_TAG_arg_variable ]
!8 = !{!"0x2e\00bar\00bar\00bar\006\000\001\000\006\000\001\006", !18, !2, !4, null, i32 (i32)* @bar, null, null, !16} ; [ DW_TAG_subprogram ]
!9 = !MDLocation(line: 3, scope: !10)
!10 = !{!"0xb\002\000\000", !18, !1} ; [ DW_TAG_lexical_block ]
!11 = !{i32 1}
!12 = !MDLocation(line: 3, scope: !10, inlinedAt: !13)
!13 = !MDLocation(line: 7, scope: !14)
!14 = !{!"0xb\006\000\000", !18, !8} ; [ DW_TAG_lexical_block ]
!15 = !{!0}
!16 = !{!7}
!17 = !{!1, !8}
!18 = !{!"f.c", !"/tmp"}
!19 = !{i32 0}

;CHECK: DEBUG_VALUE: bar:x <- E
;CHECK: Ltmp
;CHECK:	DEBUG_VALUE: foo:y <- 1{{$}}
!20 = !{i32 1, !"Debug Info Version", i32 2}
