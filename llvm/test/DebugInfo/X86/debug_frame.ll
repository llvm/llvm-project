; RUN: llc %s -mtriple=i686-pc-linux-gnu -o - | FileCheck %s

; Test that we produce a .debug_frame, not an .eh_frame

; CHECK: .cfi_sections .debug_frame

define void @f() nounwind {
entry:
  ret void
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"f", metadata !"f", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void ()* @f, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build", metadata !"clang version 3.0 ()", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
