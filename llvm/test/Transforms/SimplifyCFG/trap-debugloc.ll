; RUN: opt -S -simplifycfg < %s | FileCheck %s
; Radar 9342286
; Assign DebugLoc to trap instruction.
define void @foo() nounwind ssp {
; CHECK: call void @llvm.trap(), !dbg
  store i32 42, i32* null, !dbg !5
  ret void, !dbg !7
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, void ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"foo.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"foo.c", metadata !"/private/tmp", metadata !"Apple clang version 3.0 (tags/Apple/clang-206.1) (based on LLVM 3.0svn)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 4, i32 2, metadata !6, null}
!6 = metadata !{i32 589835, metadata !0, i32 3, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 5, i32 1, metadata !6, null}
