; RUN: opt -analyze -module-debuginfo < %s | FileCheck %s

; Produced from linking:
;   /tmp/test1.c containing f()
;   /tmp/test2.c containing g()

; Verify that both compile units and both their contained functions are
; listed by DebugInfoFinder:
;CHECK: Compile Unit: [ DW_TAG_compile_unit ] [/tmp/test1.c] [DW_LANG_C99]
;CHECK: Compile Unit: [ DW_TAG_compile_unit ] [/tmp/test2.c] [DW_LANG_C99]
;CHECK: Subprogram: [ DW_TAG_subprogram ] [line 1] [def] [f]
;CHECK: Subprogram: [ DW_TAG_subprogram ] [line 1] [def] [g]

define void @f() {
  ret void, !dbg !14
}

define void @g() {
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!13, !16}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (192092)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/test1.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test1.c", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f", metadata !"f", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @f, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/test1.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786449, metadata !9, i32 12, metadata !"clang version 3.4 (192092)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !10, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/test2.c] [DW_LANG_C99]
!9 = metadata !{metadata !"test2.c", metadata !"/tmp"}
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786478, metadata !9, metadata !12, metadata !"g", metadata !"g", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @g, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [g]
!12 = metadata !{i32 786473, metadata !9}         ; [ DW_TAG_file_type ] [/tmp/test2.c]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 1, i32 0, metadata !4, null}
!15 = metadata !{i32 1, i32 0, metadata !11, null}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
