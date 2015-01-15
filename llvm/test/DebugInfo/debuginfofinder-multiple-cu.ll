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

!0 = !{!"0x11\0012\00clang version 3.4 (192092)\000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/test1.c] [DW_LANG_C99]
!1 = !{!"test1.c", !"/tmp"}
!2 = !{i32 0}
!3 = !{!4}
!4 = !{!"0x2e\00f\00f\00\001\000\001\000\006\000\000\001", !1, !5, !6, null, void ()* @f, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/test1.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x11\0012\00clang version 3.4 (192092)\000\00\000\00\000", !9, !2, !2, !10, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/test2.c] [DW_LANG_C99]
!9 = !{!"test2.c", !"/tmp"}
!10 = !{!11}
!11 = !{!"0x2e\00g\00g\00\001\000\001\000\006\000\000\001", !9, !12, !6, null, void ()* @g, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [g]
!12 = !{!"0x29", !9}         ; [ DW_TAG_file_type ] [/tmp/test2.c]
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !MDLocation(line: 1, scope: !4)
!15 = !MDLocation(line: 1, scope: !11)
!16 = !{i32 1, !"Debug Info Version", i32 2}
