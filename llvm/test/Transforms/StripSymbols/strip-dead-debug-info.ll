; RUN: opt -strip-dead-debug-info -verify %s -S | FileCheck %s

; CHECK: ModuleID = '{{.*}}'
; CHECK-NOT: bar
; CHECK-NOT: abcd

@xyz = global i32 2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

; Function Attrs: nounwind readnone ssp
define i32 @fn() #1 {
entry:
  ret i32 0, !dbg !18
}

; Function Attrs: nounwind readonly ssp
define i32 @foo(i32 %i) #2 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !15, metadata !{}), !dbg !20
  %.0 = load i32* @xyz, align 4
  ret i32 %.0, !dbg !21
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone ssp }
attributes #2 = { nounwind readonly ssp }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25}

!0 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\001", !1, !2, !2, !23, !24, null} ; [ DW_TAG_compile_unit ] [/tmp//g.c] [DW_LANG_C89]
!1 = !{!"g.c", !"/tmp/"}
!2 = !{null}
!3 = !{!"0x2e\00bar\00bar\00\005\001\001\000\006\000\001\000", !1, null, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 5] [local] [def] [scope 0] [bar]
!4 = !{!"0x15\00\000\000\000\000\000\000", !1, !5, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp//g.c]
!6 = !{!"0x2e\00fn\00fn\00fn\006\000\001\000\006\000\001\000", !1, null, !7, null, i32 ()* @fn, null, null, null} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [fn]
!7 = !{!"0x15\00\000\000\000\000\000\000", !1, !5, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", !1, !5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x2e\00foo\00foo\00foo\007\000\001\000\006\000\001\000", !1, null, !11, null, i32 (i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 0] [foo]
!11 = !{!"0x15\00\000\000\000\000\000\000", !1, !5, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!9, !9}
!13 = !{!"0x100\00bb\005\000", !14, !5, !9} ; [ DW_TAG_auto_variable ]
!14 = !{!"0xb\005\000\000", !1, !3} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!15 = !{!"0x101\00i\007\000", !10, !5, !9} ; [ DW_TAG_arg_variable ]
!16 = !{!"0x34\00abcd\00abcd\00\002\001\001", !5, !5, !9, null, null} ; [ DW_TAG_variable ]
!17 = !{!"0x34\00xyz\00xyz\00\003\000\001", !5, !5, !9, i32* @xyz, null} ; [ DW_TAG_variable ]
!18 = !MDLocation(line: 6, scope: !19)
!19 = !{!"0xb\006\000\000", !1, !6} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!20 = !MDLocation(line: 7, scope: !10)
!21 = !MDLocation(line: 10, scope: !22)
!22 = !{!"0xb\007\000\000", !1, !10} ; [ DW_TAG_lexical_block ] [/tmp//g.c]
!23 = !{!3, !6, !10}
!24 = !{!16, !17}
!25 = !{i32 1, !"Debug Info Version", i32 2}
