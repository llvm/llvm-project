; RUN: llc < %s - | FileCheck %s
;
; FIXME: A potentially more interesting test case would be:
; %call = @bar()
; dbg.value j=0
; %call2 = @bar()
; dbg.value j=%call
;
; We cannot current handle the above sequence because codegenprepare
; hoists the second dbg.value above %call2, which then appears to
; conflict with j=0. It does this because SelectionDAG cannot handle
; global debug values.

target triple = "x86_64-apple-darwin10.0.0"

;CHECK:        ## DW_OP_consts
;CHECK-NEXT:  .byte	42
define i32 @foobar() nounwind readonly noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !9
  %call = tail call i32 @bar(), !dbg !11
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !11
  %call2 = tail call i32 @bar(), !dbg !11
  %add = add nsw i32 %call2, %call, !dbg !12
  ret i32 %add, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
declare i32 @bar() nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = !{!"0x2e\00foobar\00foobar\00foobar\0012\000\001\000\006\000\001\000", !15, !1, !3, null, i32 ()* @foobar, null, null, !14} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !15} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 114183)\001\00\000\00\001", !15, !16, !16, !13, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !15, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !15, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x100\00j\0015\000", !7, !1, !5} ; [ DW_TAG_auto_variable ]
!7 = !{!"0xb\0012\0052\000", !15, !0} ; [ DW_TAG_lexical_block ]
!8 = !{i32 42}
!9 = !MDLocation(line: 15, column: 12, scope: !7)
!10 = !MDLocation(line: 23, column: 3, scope: !7)
!11 = !MDLocation(line: 17, column: 3, scope: !7)
!12 = !MDLocation(line: 18, column: 3, scope: !7)
!13 = !{!0}
!14 = !{!6}
!15 = !{!"mu.c", !"/private/tmp"}
!16 = !{i32 0}
!17 = !{i32 1, !"Debug Info Version", i32 2}
