; RUN: llc -mtriple x86_64-apple-darwin10.0.0 -disable-cfi %s -o - | FileCheck %s

define i32 @f() nounwind {
entry:
  ret i32 42
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}
!6 = metadata !{metadata !0}

!0 = metadata !{i32 786478, metadata !7, metadata !1, metadata !"f", metadata !"f", metadata !"", i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @f, null, null, null, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!1 = metadata !{i32 786473, metadata !7} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !7, i32 12, metadata !"clang version 3.0 ()", i1 true, metadata !"", i32 0, metadata !8, metadata !8, metadata !6, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !7, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build-rust2"}
!8 = metadata !{i32 0}

; CHECK:      _f:                                     ## @f
; CHECK-NEXT: Ltmp0:

; CHECK:      Ltmp9 = (Ltmp3-Ltmp2)-0
; CHECK-NEXT:	.long	Ltmp9
; CHECK-NEXT:	.quad	Ltmp0
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
