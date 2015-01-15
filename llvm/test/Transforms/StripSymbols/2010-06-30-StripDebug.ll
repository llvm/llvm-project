; RUN: opt -strip-debug < %s -S | FileCheck %s

; CHECK-NOT: llvm.dbg

@x = common global i32 0                          ; <i32*> [#uses=0]

define void @foo() nounwind readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !5, metadata !{}), !dbg !10
  ret void, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}
!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.foo = !{!5}
!llvm.dbg.gv = !{!8}

!0 = !{!"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\001\000", !12, !1, !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !12} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !12, !4, !4, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !12, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x100\00y\003\000", !6, !1, !7} ; [ DW_TAG_auto_variable ]
!6 = !{!"0xb\002\000\000", !12, !0} ; [ DW_TAG_lexical_block ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", !12, !1} ; [ DW_TAG_base_type ]
!8 = !{!"0x34\00x\00x\00\001\000\001", !1, !1, !7, i32* @x} ; [ DW_TAG_variable ]
!9 = !{i32 0}
!10 = !MDLocation(line: 3, scope: !6)
!11 = !MDLocation(line: 4, scope: !6)
!12 = !{!"b.c", !"/tmp"}
!13 = !{i32 1, !"Debug Info Version", i32 2}
