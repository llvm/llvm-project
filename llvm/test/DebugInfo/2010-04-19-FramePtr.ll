; RUN: llc -asm-verbose -O1 -o %t < %s 
; RUN: grep DW_AT_APPLE_omit_frame_ptr %t
; RUN: llc -disable-fp-elim -asm-verbose -O1 -o %t < %s 
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
!9 = metadata !{metadata !1}

!0 = metadata !{i32 2, i32 0, metadata !1, null}
!1 = metadata !{i32 786478, metadata !10, null, metadata !"foo", metadata !"foo", metadata !"foo", i32 2, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @foo, null, null, null, i32 2} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !10} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !10, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 false, metadata !"", i32 0, metadata !11, metadata !11, metadata !9, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !10, metadata !2, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786468, metadata !10, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 2, i32 0, metadata !8, null}
!8 = metadata !{i32 786443, metadata !10, metadata !1, i32 2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"a.c", metadata !"/tmp"}
!11 = metadata !{i32 0}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
