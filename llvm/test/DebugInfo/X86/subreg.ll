; RUN: llc %s -mtriple=x86_64-pc-linux-gnu -O0 -o - | FileCheck %s

; We are testing that a value in a 16 bit register gets reported as
; being in its superregister.
; FIXME: There should be a DW_OP_bit_piece too.

; CHECK: .byte   80                      # DW_OP_reg0

define i16 @f(i16 signext %zzz) nounwind {
entry:
  call void @llvm.dbg.value(metadata !{i16 %zzz}, i64 0, metadata !0)
  %conv = sext i16 %zzz to i32, !dbg !7
  %conv1 = trunc i32 %conv to i16
  ret i16 %conv1
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!11}
!9 = metadata !{metadata !1}

!0 = metadata !{i32 786689, metadata !1, metadata !"zzz", metadata !2, i32 16777219, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 786478, metadata !10, metadata !2, metadata !"f", metadata !"f", metadata !"", i32 3, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i16 (i16)* @f, null, null, null, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [f]
!2 = metadata !{i32 786473, metadata !10} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !10, i32 12, metadata !"clang version 3.0 ()", i1 false, metadata !"", i32 0, metadata !5, metadata !5, metadata !9, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !10, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{null}
!6 = metadata !{i32 786468, null, metadata !3, metadata !"short", i32 0, i64 16, i64 16, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 4, i32 22, metadata !8, null}
!8 = metadata !{i32 786443, metadata !10, metadata !1, i32 3, i32 19, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/tmpfs/build"}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
