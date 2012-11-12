; RUN: llc -mtriple=i386-apple-macosx -disable-cfi %s -o - | FileCheck %s

; test that we don't produce foo.eh symbols is a debug_frame section.
; CHECK-NOT: .globl	_f.eh

define i32 @f() nounwind readnone optsize {
entry:
  ret i32 42
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"f", metadata !"f", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 ()* @f, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/tmpfs/build", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/tmpfs/build", metadata !"clang version 3.0 ()", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
