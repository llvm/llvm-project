; RUN: opt -instcombine -S < %s | FileCheck %s


@.str = private constant [3 x i8] c"%c\00"

define void @foo() nounwind ssp {
;CHECK: call i32 @putchar{{.+}} !dbg
  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 97), !dbg !5
  ret void, !dbg !7
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, metadata !8, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !8} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !8, i32 12, metadata !"clang", i1 true, metadata !"", i32 0, metadata !4, metadata !4, metadata !9, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !8, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 5, i32 2, metadata !6, null}
!6 = metadata !{i32 589835, metadata !8, metadata !0, i32 4, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 6, i32 1, metadata !6, null}
!8 = metadata !{metadata !"m.c", metadata !"/private/tmp"}
!9 = metadata !{metadata !0}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
