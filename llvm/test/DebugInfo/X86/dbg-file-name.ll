; RUN: llc -enable-dwarf-directory -mtriple x86_64-apple-darwin10.0.0  < %s | FileCheck %s

; Radar 8884898
; CHECK: file	1 "simple.c"

declare i32 @printf(i8*, ...) nounwind

define i32 @main() nounwind {
  ret i32 0
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12}

!1 = metadata !{i32 786473, metadata !10} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !10, i32 1, metadata !"LLVM build 00", i1 true, metadata !"", i32 0, metadata !11, metadata !11, metadata !9, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 786468, metadata !10, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786478, metadata !10, metadata !1, metadata !"main", metadata !"main", metadata !"main", i32 9, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, null, i32 0} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 786453, metadata !10, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !5}
!9 = metadata !{metadata !6}
!10 = metadata !{metadata !"simple.c", metadata !"/Users/manav/one/two"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
