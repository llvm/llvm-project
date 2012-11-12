; RUN: opt -simplify-libcalls -S < %s | FileCheck %s


@.str = private constant [3 x i8] c"%c\00"

define void @foo() nounwind ssp {
;CHECK: call i32 @putchar{{.+}} !dbg
  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 97), !dbg !5
  ret void, !dbg !7
}

declare i32 @printf(i8*, ...)

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"m.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"m.c", metadata !"/private/tmp", metadata !"clang", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 5, i32 2, metadata !6, null}
!6 = metadata !{i32 589835, metadata !0, i32 4, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!7 = metadata !{i32 6, i32 1, metadata !6, null}

