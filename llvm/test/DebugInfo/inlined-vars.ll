; RUN: llc -O0 < %s | FileCheck %s -check-prefix ARGUMENT
; RUN: llc -O0 < %s | FileCheck %s -check-prefix VARIABLE
; PR 13202

define i32 @main() uwtable {
entry:
  tail call void @llvm.dbg.value(metadata !1, i64 0, metadata !18), !dbg !21
  tail call void @llvm.dbg.value(metadata !1, i64 0, metadata !22), !dbg !23
  tail call void @smth(i32 0), !dbg !24
  tail call void @smth(i32 0), !dbg !25
  ret i32 0, !dbg !19
}

declare void @smth(i32)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = metadata !{i32 786449, metadata !26, i32 4, metadata !"clang version 3.2 (trunk 159419)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !10}
!5 = metadata !{i32 786478, metadata !26, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 10, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @main, null, null, metadata !1, i32 10} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !26} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786478, metadata !26, metadata !6, metadata !"f", metadata !"f", metadata !"_ZL1fi", i32 3, metadata !11, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, null, metadata !13, i32 3} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 786453, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !9, metadata !9}
!13 = metadata !{metadata !14}
!14 = metadata !{metadata !15, metadata !16}
!15 = metadata !{i32 786689, metadata !10, metadata !"argument", metadata !6, i32 16777219, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]

; Two DW_TAG_formal_parameter: one abstract and one inlined.
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT: {{.*Abbrev.*DW_TAG_formal_parameter}}
; ARGUMENT-NOT: {{.*Abbrev.*DW_TAG_formal_parameter}}

!16 = metadata !{i32 786688, metadata !10, metadata !"local", metadata !6, i32 4, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]

; Two DW_TAG_variable: one abstract and one inlined.
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE: {{.*Abbrev.*DW_TAG_variable}}
; VARIABLE-NOT: {{.*Abbrev.*DW_TAG_variable}}

!18 = metadata !{i32 786689, metadata !10, metadata !"argument", metadata !6, i32 16777219, metadata !9, i32 0, metadata !19} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 11, i32 10, metadata !5, null}
!21 = metadata !{i32 3, i32 25, metadata !10, metadata !19}
!22 = metadata !{i32 786688, metadata !10, metadata !"local", metadata !6, i32 4, metadata !9, i32 0, metadata !19} ; [ DW_TAG_auto_variable ]
!23 = metadata !{i32 4, i32 16, metadata !10, metadata !19}
!24 = metadata !{i32 5, i32 3, metadata !10, metadata !19}
!25 = metadata !{i32 6, i32 3, metadata !10, metadata !19}
!26 = metadata !{metadata !"inline-bug.cc", metadata !"/tmp/dbginfo/pr13202"}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
