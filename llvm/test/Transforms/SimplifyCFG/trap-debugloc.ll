; RUN: opt -S -simplifycfg < %s | FileCheck %s
; Radar 9342286
; Assign DebugLoc to trap instruction.
define void @foo() nounwind ssp {
; CHECK: call void @llvm.trap(), !dbg
  store i32 42, i32* null, !dbg !5
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!llvm.dbg.sp = !{!0}

!0 = !{!"0x2e\00foo\00foo\00\003\000\001\000\006\000\000\000", !8, !1, !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [foo]
!1 = !{!"0x29", !8} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00Apple clang version 3.0 (tags/Apple/clang-206.1) (based on LLVM 3.0svn)\001\00\000\00\000", !8, !4, !4, !9, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !8, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !MDLocation(line: 4, column: 2, scope: !6)
!6 = !{!"0xb\003\0012\000", !8, !0} ; [ DW_TAG_lexical_block ]
!7 = !MDLocation(line: 5, column: 1, scope: !6)
!8 = !{!"foo.c", !"/private/tmp"}
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 2}
