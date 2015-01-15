; RUN: %llc_dwarf -asm-verbose -O1 -o %t < %s
; RUN: grep DW_AT_APPLE_omit_frame_ptr %t
; RUN: %llc_dwarf -disable-fp-elim -asm-verbose -O1 -o %t < %s
; RUN: grep -v DW_AT_APPLE_omit_frame_ptr %t


define i32 @foo() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 42, i32* %0, align 4, !dbg !0
  %1 = load i32* %0, align 4, !dbg !0             ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !0
  br label %return, !dbg !0

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !0           ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !7
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!12}
!9 = !{!1}

!0 = !MDLocation(line: 2, scope: !1)
!1 = !{!"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\000\002", !10, null, !4, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !10} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !10, !11, !11, !9, null,  null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !10, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", !10, !2} ; [ DW_TAG_base_type ]
!7 = !MDLocation(line: 2, scope: !8)
!8 = !{!"0xb\002\000\000", !10, !1} ; [ DW_TAG_lexical_block ]
!10 = !{!"a.c", !"/tmp"}
!11 = !{i32 0}
!12 = !{i32 1, !"Debug Info Version", i32 2}
