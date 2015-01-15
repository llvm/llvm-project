; RUN: llc  -o /dev/null < %s
; Radar 7937664
%struct.AppleEvent = type opaque

define void @DisposeDMNotificationUPP(void (%struct.AppleEvent*)* %userUPP) "no-frame-pointer-elim-non-leaf" nounwind ssp {
entry:
  %userUPP_addr = alloca void (%struct.AppleEvent*)* ; <void (%struct.AppleEvent*)**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata void (%struct.AppleEvent*)** %userUPP_addr, metadata !0, metadata !{!"0x102"}), !dbg !13
  store void (%struct.AppleEvent*)* %userUPP, void (%struct.AppleEvent*)** %userUPP_addr
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!19}
!0 = !{!"0x101\00userUPP\007\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00DisposeDMNotificationUPP\00DisposeDMNotificationUPP\00DisposeDMNotificationUPP\007\000\001\000\006\000\000\000", !16, null, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !16} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)\001\00\000\00\000", !16, !17, !17, !18, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !16, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{null, !6}
!6 = !{!"0x16\00DMNotificationUPP\006\000\000\000\000", !16, !2, !7} ; [ DW_TAG_typedef ]
!7 = !{!"0xf\00\000\0064\0064\000\000", !16, !2, !8} ; [ DW_TAG_pointer_type ]
!8 = !{!"0x15\00\000\000\000\000\000\000", !16, !2, null, !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = !{null, !10}
!10 = !{!"0xf\00\000\0064\0064\000\000", !16, !2, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x16\00AppleEvent\004\000\000\000\000", !16, !2, !12} ; [ DW_TAG_typedef ]
!12 = !{!"0x13\00AEDesc\001\000\000\000\004\000", !16, !2, null, null, null, null, null} ; [ DW_TAG_structure_type ] [AEDesc] [line 1, size 0, align 0, offset 0] [decl] [from ]
!13 = !MDLocation(line: 7, scope: !1)
!14 = !MDLocation(line: 8, scope: !15)
!15 = !{!"0xb\007\000\000", !16, !1} ; [ DW_TAG_lexical_block ]
!16 = !{!"t.c", !"/Users/echeng/LLVM/radars/r7937664/"}
!17 = !{i32 0}
!18 = !{!1}
!19 = !{i32 1, !"Debug Info Version", i32 2}
