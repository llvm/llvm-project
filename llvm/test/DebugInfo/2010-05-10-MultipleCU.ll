; RUN: llc -O0 -asm-verbose < %s | FileCheck %s
; One for a.c, second one for b.c and third one for abbrev.

; CHECK: info_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: info_end

; CHECK: info_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: info_end

; CHECK: abbrev_begin
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG_compile_unit
; CHECK: abbrev_end

define i32 @foo() nounwind readnone ssp {
return:
  ret i32 42, !dbg !0
}

define i32 @bar() nounwind readnone ssp {
return:
  ret i32 21, !dbg !8
}

!llvm.dbg.cu = !{!4, !12}
!llvm.module.flags = !{!21}
!16 = metadata !{metadata !2}
!17 = metadata !{metadata !10}

!0 = metadata !{i32 3, i32 0, metadata !1, null}
!1 = metadata !{i32 786443, metadata !18, metadata !2, i32 2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 786478, metadata !18, metadata !3, metadata !"foo", metadata !"foo", metadata !"foo", i32 2, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @foo, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 786473, metadata !18} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 786449, metadata !18, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 false, metadata !"", i32 0, metadata !19, metadata !19, metadata !16, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 786453, metadata !18, metadata !3, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786468, metadata !18, metadata !3, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 3, i32 0, metadata !9, null}
!9 = metadata !{i32 786443, metadata !20, metadata !10, i32 2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 786478, metadata !20, metadata !11, metadata !"bar", metadata !"bar", metadata !"bar", i32 2, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @bar, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 786473, metadata !20} ; [ DW_TAG_file_type ]
!12 = metadata !{i32 786449, metadata !20, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 false, metadata !"", i32 0, metadata !19, metadata !19, metadata !17, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!13 = metadata !{i32 786453, metadata !20, metadata !11, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786468, metadata !20, metadata !11, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !"a.c", metadata !"/tmp/"}
!19 = metadata !{i32 0}
!20 = metadata !{metadata !"b.c", metadata !"/tmp/"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
