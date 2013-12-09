; RUN: llc -mtriple=i386-apple-macosx -disable-cfi %s -o - | FileCheck %s

; test that we don't produce foo.eh symbols is a debug_frame section.
; CHECK-NOT: .globl	_f.eh

define i32 @f() nounwind readnone optsize {
entry:
  ret i32 42
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}
!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, metadata !6, metadata !1, metadata !"f", metadata !"f", metadata !"", i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @f, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [f]
!1 = metadata !{i32 589865, metadata !6} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, metadata !6, i32 12, metadata !"clang version 3.0 ()", i1 true, metadata !"", i32 0, metadata !7, metadata !7, metadata !8, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !6, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/tmpfs/build"}
!7 = metadata !{i32 0}
!8 = metadata !{metadata !0}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
