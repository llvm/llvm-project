; RUN: llc  -o /dev/null -disable-non-leaf-fp-elim < %s
; Radar 7937664
%struct.AppleEvent = type opaque

define void @DisposeDMNotificationUPP(void (%struct.AppleEvent*)* %userUPP) nounwind ssp {
entry:
  %userUPP_addr = alloca void (%struct.AppleEvent*)* ; <void (%struct.AppleEvent*)**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{void (%struct.AppleEvent*)** %userUPP_addr}, metadata !0), !dbg !13
  store void (%struct.AppleEvent*)* %userUPP, void (%struct.AppleEvent*)** %userUPP_addr
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 524545, metadata !1, metadata !"userUPP", metadata !2, i32 7, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"DisposeDMNotificationUPP", metadata !"DisposeDMNotificationUPP", metadata !"DisposeDMNotificationUPP", metadata !2, i32 7, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"t.c", metadata !"/Users/echeng/LLVM/radars/r7937664/", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"t.c", metadata !"/Users/echeng/LLVM/radars/r7937664/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{null, metadata !6}
!6 = metadata !{i32 524310, metadata !2, metadata !"DMNotificationUPP", metadata !2, i32 6, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_typedef ]
!7 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0, null} ; [ DW_TAG_subroutine_type ]
!9 = metadata !{null, metadata !10}
!10 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 524310, metadata !2, metadata !"AppleEvent", metadata !2, i32 4, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_typedef ]
!12 = metadata !{i32 524307, metadata !2, metadata !"AEDesc", metadata !2, i32 1, i64 0, i64 0, i64 0, i32 4, null, null, i32 0, null} ; [ DW_TAG_structure_type ]
!13 = metadata !{i32 7, i32 0, metadata !1, null}
!14 = metadata !{i32 8, i32 0, metadata !15, null}
!15 = metadata !{i32 524299, metadata !1, i32 7, i32 0} ; [ DW_TAG_lexical_block ]
