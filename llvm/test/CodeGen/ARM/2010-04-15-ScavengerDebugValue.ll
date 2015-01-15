; RUN: llc < %s
; PR6847
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv4t-apple-darwin10"

define hidden i32 @__addvsi3(i32 %a, i32 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !0, metadata !{!"0x102"})
  %0 = add nsw i32 %b, %a, !dbg !9                ; <i32> [#uses=1]
  ret i32 %0, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!15}
!0 = !{!"0x101\00b\0093\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00__addvsi3\00__addvsi3\00__addvsi3\0094\000\001\000\006\000\000\000", !12, null, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!12 = !{!"libgcc2.c", !"/Users/bwilson/local/nightly/test-2010-04-14/build/llvmgcc.roots/llvmgcc~obj/src/gcc"}
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 00)\001\00\000\00\000", !12, !13, !13, !14, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !12, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !6, !6}
!6 = !{!"0x16\00SItype\00152\000\000\000\000", !12, null, !8} ; [ DW_TAG_typedef ]
!7 = !{!"0x29", !"libgcc2.h", !"/Users/bwilson/local/nightly/test-2010-04-14/build/llvmgcc.roots/llvmgcc~obj/src/gcc", !3} ; [ DW_TAG_file_type ]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", !12, !2} ; [ DW_TAG_base_type ]
!9 = !MDLocation(line: 95, scope: !10)
!10 = !{!"0xb\0094\000\000", !12, !1} ; [ DW_TAG_lexical_block ]
!11 = !MDLocation(line: 100, scope: !10)
!13 = !{i32 0}
!14 = !{!1}
!15 = !{i32 1, !"Debug Info Version", i32 2}
