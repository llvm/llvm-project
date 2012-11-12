; RUN: llc -mtriple x86_64-apple-darwin10.0.0 -disable-cfi %s -o - | FileCheck %s

define i32 @f() nounwind {
entry:
  ret i32 42
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"f", metadata !"f", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 ()* @f, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build-rust2", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build-rust2", metadata !"clang version 3.0 ()", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]


; CHECK:      _f:                                     ## @f
; CHECK-NEXT: Ltmp0:

; CHECK:      Ltmp9 = (Ltmp3-Ltmp2)-0
; CHECK-NEXT:	.long	Ltmp9
; CHECK-NEXT:	.quad	Ltmp0
