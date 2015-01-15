; For http://llvm.org/bugs/show_bug.cgi?id=12942
;   There are two CUs coming from /tmp/foo.c in this module. Make sure it doesn't
;   blow llc up and produces something reasonable.
;

; REQUIRES: object-emission

; RUN: %llc_dwarf %s -o %t -filetype=obj -O0
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
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !22, metadata !{!"0x102"}), !dbg !27
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @str1, i32 0, i32 0)), !dbg !28
  tail call void @foo() nounwind, !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!33}

!0 = !{!"0x11\0012\00clang version 3.2 (trunk 156513)\001\00\000\00\001", !32, !1, !1, !3, !1, !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00\005\000\001\000\006\00256\001\005", !32, !6, !7, null, void ()* @foo, null, null, !1} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !32} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!"0x11\0012\00clang version 3.2 (trunk 156513)\001\00\000\00\001", !32, !1, !1, !10, !1, !1} ; [ DW_TAG_compile_unit ]
!10 = !{!12}
!12 = !{!"0x2e\00main\00main\00\0011\000\001\000\006\00256\001\0011", !32, !6, !13, null, i32 (i32, i8**)* @main, null, null, !19} ; [ DW_TAG_subprogram ]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!15, !15, !16}
!15 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!16 = !{!"0xf\00\000\0032\0032\000\000", null, null, !17} ; [ DW_TAG_pointer_type ]
!17 = !{!"0xf\00\000\0032\0032\000\000", null, null, !18} ; [ DW_TAG_pointer_type ]
!18 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!19 = !{!21, !22}
!21 = !{!"0x101\00argc\0016777227\000", !12, !6, !15} ; [ DW_TAG_arg_variable ]
!22 = !{!"0x101\00argv\0033554443\000", !12, !6, !16} ; [ DW_TAG_arg_variable ]
!23 = !MDLocation(line: 6, column: 3, scope: !24)
!24 = !{!"0xb\005\0016\000", !32, !5} ; [ DW_TAG_lexical_block ]
!25 = !MDLocation(line: 7, column: 1, scope: !24)
!26 = !MDLocation(line: 11, column: 14, scope: !12)
!27 = !MDLocation(line: 11, column: 26, scope: !12)
!28 = !MDLocation(line: 12, column: 3, scope: !29)
!29 = !{!"0xb\0011\0034\000", !32, !12} ; [ DW_TAG_lexical_block ]
!30 = !MDLocation(line: 13, column: 3, scope: !29)
!31 = !MDLocation(line: 14, column: 3, scope: !29)
!32 = !{!"foo.c", !"/tmp"}

; This test is simple to be cross platform (many targets don't yet have
; sufficiently good DWARF emission and/or dumping)
; CHECK: {{DW_TAG_compile_unit}}
; CHECK: {{foo\.c}}

!33 = !{i32 1, !"Debug Info Version", i32 2}
