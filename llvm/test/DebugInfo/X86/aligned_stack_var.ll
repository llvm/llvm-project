; RUN: llc %s -mtriple=x86_64-pc-linux-gnu -O0 -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; If stack is realigned, we shouldn't describe locations of local
; variables by giving offset from the frame pointer (%rbp):
; push %rpb
; mov  %rsp,%rbp
; and  ALIGNMENT,%rsp ; (%rsp and %rbp are different now)
; It's better to use offset from %rsp instead.

; DW_AT_location of variable "x" shouldn't be equal to
; (DW_OP_fbreg: .*): DW_OP_fbreg has code 0x91

; CHECK: {{0x.* DW_TAG_variable}}
; CHECK-NOT: {{DW_AT_location.*DW_FORM_block1.*0x.*91}}
; CHECK: NULL

define void @_Z3runv() nounwind uwtable {
entry:
  %x = alloca i32, align 32
  call void @llvm.dbg.declare(metadata !{i32* %x}, metadata !9), !dbg !12
  ret void, !dbg !13
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = metadata !{i32 786449, metadata !14, i32 4, metadata !"clang version 3.2 (trunk 155696:155697) (llvm/trunk 155696)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !14, metadata !6, metadata !"run", metadata !"run", metadata !"_Z3runv", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3runv, null, null, metadata !1, i32 1} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !14} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{i32 786688, metadata !10, metadata !"x", metadata !6, i32 2, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 786443, metadata !14, metadata !5, i32 1, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!12 = metadata !{i32 2, i32 7, metadata !10, null}
!13 = metadata !{i32 3, i32 1, metadata !10, null}
!14 = metadata !{metadata !"test.cc", metadata !"/home/samsonov/debuginfo"}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
