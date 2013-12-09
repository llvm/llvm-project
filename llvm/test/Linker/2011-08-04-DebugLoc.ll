; RUN: llvm-link %s %p/2011-08-04-DebugLoc2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if DebugLoc is mapped properly or not.


;CHECK-NOT: metadata !{i32 589870, i32 0, metadata !{{[0-9]+}}, metadata !"bar", metadata !"bar", metadata !"", metadata !{{[0-9]+}}, i32 1, metadata !{{[0-9]+}}, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, null, null, null}


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @foo() nounwind ssp {
  ret i32 42, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}

!0 = metadata !{i32 589841, metadata !8, i32 12, metadata !"Apple clang version 3.0 (tags/Apple/clang-209.11) (based on LLVM 3.0svn)", i1 true, metadata !"", i32 0, metadata !9, metadata !9, metadata !10, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, metadata !8, metadata !2, metadata !"foo", metadata !"foo", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = metadata !{i32 589865, metadata !8} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !8, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 2, i32 13, metadata !7, null}
!7 = metadata !{i32 589835, metadata !8, metadata !1, i32 2, i32 11, i32 0} ; [ DW_TAG_lexical_block ]
!8 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!9 = metadata !{i32 0}
!10 = metadata !{metadata !1}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
