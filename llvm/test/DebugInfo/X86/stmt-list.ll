; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

; CHECK:      .section        .debug_line,"",@progbits
; CHECK-NEXT: .Lsection_line:

; CHECK:      .long   .Lsection_line          # DW_AT_stmt_list

define void @f() {
entry:
  ret void
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"f", metadata !"f", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void ()* @f, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"test2.c", metadata !"/home/espindola/llvm", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"test2.c", metadata !"/home/espindola/llvm", metadata !"clang version 3.0 ()", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
