; This file is used by 2011-08-04-Metadata.ll, so it doesn't actually do anything itself
;
; RUN: true


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = internal global i32 0, align 4

define void @bar() nounwind uwtable ssp {
entry:
  store i32 1, i32* @x, align 4, !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}
!llvm.dbg.gv = !{!5}

!0 = metadata !{i32 589841, metadata !9, i32 12, metadata !"clang version 3.0 ()", i1 true, metadata !"", i32 0, metadata !4, metadata !4, metadata !10, null, null, metadata !""}
!1 = metadata !{i32 589870, metadata !9, metadata !2, metadata !"bar", metadata !"bar", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @bar, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [bar]
!2 = metadata !{i32 589865, metadata !9}
!3 = metadata !{i32 589845, metadata !9, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 589876, i32 0, metadata !0, metadata !"x", metadata !"x", metadata !"", metadata !2, i32 1, metadata !6, i32 1, i32 1, i32* @x}
!6 = metadata !{i32 589860, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!7 = metadata !{i32 2, i32 14, metadata !8, null}
!8 = metadata !{i32 589835, metadata !9, metadata !1, i32 2, i32 12, i32 0}
!9 = metadata !{metadata !"/tmp/two.c", metadata !"/Volumes/Lalgate/Slate/D"}
!10 = metadata !{metadata !1}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
