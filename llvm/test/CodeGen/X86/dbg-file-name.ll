; RUN: llc -enable-dwarf-directory -mtriple x86_64-apple-darwin10.0.0  < %s | FileCheck %s

; Radar 8884898
; CHECK: file	1 "simple.c"

declare i32 @printf(i8*, ...) nounwind

define i32 @main() nounwind {
  ret i32 0
}

!llvm.dbg.sp = !{ !6}

!1 = metadata !{i32 589865, metadata !"simple.c", metadata !"/Users/manav/one/two", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"simple.c", metadata !"/Users/manav/one/two", metadata !"LLVM build 00", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 589860, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 9, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !5}
