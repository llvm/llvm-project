; For http://llvm.org/bugs/show_bug.cgi?id=12942
;   There are two CUs coming from /tmp/foo.c in this module. Make sure it doesn't
;   blow llc up and produces something reasonable.
;

; REQUIRES: object-emission

; RUN: llc %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.bc'

@str = private unnamed_addr constant [4 x i8] c"FOO\00"
@str1 = private unnamed_addr constant [6 x i8] c"Main!\00"

define void @foo() nounwind {
entry:
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8]* @str, i32 0, i32 0)), !dbg !23
  ret void, !dbg !25
}

declare i32 @puts(i8* nocapture) nounwind

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !21), !dbg !26
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !22), !dbg !27
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @str1, i32 0, i32 0)), !dbg !28
  tail call void @foo() nounwind, !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!33}

!0 = metadata !{i32 786449, metadata !32, i32 12, metadata !"clang version 3.2 (trunk 156513)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1, metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !32, metadata !6, metadata !"foo", metadata !"foo", metadata !"", i32 5, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @foo, null, null, metadata !1, i32 5} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !32} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{i32 786449, metadata !32, i32 12, metadata !"clang version 3.2 (trunk 156513)", i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !10, metadata !1, metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!10 = metadata !{metadata !12}
!12 = metadata !{i32 786478, metadata !32, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 11, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i8**)* @main, null, null, metadata !19, i32 11} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !15, metadata !15, metadata !16}
!15 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!16 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !17} ; [ DW_TAG_pointer_type ]
!17 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !18} ; [ DW_TAG_pointer_type ]
!18 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!19 = metadata !{metadata !20}
!20 = metadata !{metadata !21, metadata !22}
!21 = metadata !{i32 786689, metadata !12, metadata !"argc", metadata !6, i32 16777227, metadata !15, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 786689, metadata !12, metadata !"argv", metadata !6, i32 33554443, metadata !16, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 6, i32 3, metadata !24, null}
!24 = metadata !{i32 786443, metadata !32, metadata !5, i32 5, i32 16, i32 0} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 7, i32 1, metadata !24, null}
!26 = metadata !{i32 11, i32 14, metadata !12, null}
!27 = metadata !{i32 11, i32 26, metadata !12, null}
!28 = metadata !{i32 12, i32 3, metadata !29, null}
!29 = metadata !{i32 786443, metadata !32, metadata !12, i32 11, i32 34, i32 0} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 13, i32 3, metadata !29, null}
!31 = metadata !{i32 14, i32 3, metadata !29, null}
!32 = metadata !{metadata !"foo.c", metadata !"/tmp"}

; This test is simple to be cross platform (many targets don't yet have
; sufficiently good DWARF emission and/or dumping)
; CHECK: {{DW_TAG_compile_unit}}
; CHECK: {{foo\.c}}

!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
