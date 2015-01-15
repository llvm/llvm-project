; RUN: %llc_dwarf -O0 < %s | FileCheck %s -check-prefix ARGUMENT
; RUN: %llc_dwarf -O0 < %s | FileCheck %s -check-prefix VARIABLE
; PR 13202

define i32 @main() uwtable {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !22, metadata !{!"0x102"}), !dbg !23
  tail call void @smth(i32 0), !dbg !24
  tail call void @smth(i32 0), !dbg !25
  ret i32 0, !dbg !19
}

declare void @smth(i32)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = !{!"0x11\004\00clang version 3.2 (trunk 159419)\001\00\000\00\000", !26, !2, !2, !3, !2,  !2} ; [ DW_TAG_compile_unit ]
!1 = !{i32 0}
!2 = !{}
!3 = !{!5, !10}
!5 = !{!"0x2e\00main\00main\00\0010\000\001\000\006\00256\001\0010", !26, !6, !7, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !26} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0x2e\00f\00f\00_ZL1fi\003\001\001\000\006\00256\001\003", !26, !6, !11, null, null, null, null, !13} ; [ DW_TAG_subprogram ]
!11 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!9, !9}
!13 = !{!15, !16}
!15 = !{!"0x101\00argument\0016777219\000", !10, !6, !9} ; [ DW_TAG_arg_variable ]

; Two DW_TAG_formal_parameter: one abstract and one inlined.
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT-NOT: {{.*Abbrev.*DW_TAG_formal_parameter}}

!16 = !{!"0x100\00local\004\000", !10, !6, !9} ; [ DW_TAG_auto_variable ]

; Two DW_TAG_variable: one abstract and one inlined.
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE-NOT: {{.*Abbrev.*DW_TAG_variable}}

!18 = !{!"0x101\00argument\0016777219\000", !10, !6, !9, !19} ; [ DW_TAG_arg_variable ]
!19 = !MDLocation(line: 11, column: 10, scope: !5)
!21 = !MDLocation(line: 3, column: 25, scope: !10, inlinedAt: !19)
!22 = !{!"0x100\00local\004\000", !10, !6, !9, !19} ; [ DW_TAG_auto_variable ]
!23 = !MDLocation(line: 4, column: 16, scope: !10, inlinedAt: !19)
!24 = !MDLocation(line: 5, column: 3, scope: !10, inlinedAt: !19)
!25 = !MDLocation(line: 6, column: 3, scope: !10, inlinedAt: !19)
!26 = !{!"inline-bug.cc", !"/tmp/dbginfo/pr13202"}
!27 = !{i32 1, !"Debug Info Version", i32 2}
