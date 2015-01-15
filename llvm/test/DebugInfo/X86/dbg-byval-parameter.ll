; RUN: %llc_dwarf -march=x86 -asm-verbose < %s | grep DW_TAG_formal_parameter


%struct.Pt = type { double, double }
%struct.Rect = type { %struct.Pt, %struct.Pt }

define double @foo(%struct.Rect* byval %my_r0) nounwind ssp {
entry:
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.Rect* %my_r0, metadata !0, metadata !{!"0x102"}), !dbg !15
  %1 = getelementptr inbounds %struct.Rect* %my_r0, i32 0, i32 0, !dbg !16 ; <%struct.Pt*> [#uses=1]
  %2 = getelementptr inbounds %struct.Pt* %1, i32 0, i32 0, !dbg !16 ; <double*> [#uses=1]
  %3 = load double* %2, align 8, !dbg !16         ; <double> [#uses=1]
  store double %3, double* %0, align 8, !dbg !16
  %4 = load double* %0, align 8, !dbg !16         ; <double> [#uses=1]
  store double %4, double* %retval, align 8, !dbg !16
  br label %return, !dbg !16

return:                                           ; preds = %entry
  %retval1 = load double* %retval, !dbg !16       ; <double> [#uses=1]
  ret double %retval1, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!21}

!0 = !{!"0x101\00my_r0\0011\000", !1, !2, !7} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foo\00foo\00foo\0011\000\001\000\006\000\000\000", !19, !2, !4, null, double (%struct.Rect*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !19} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !19, !20, !20, !18, null,  null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !19, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !7}
!6 = !{!"0x24\00double\000\0064\0064\000\000\004", !19, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0x13\00Rect\006\00256\0064\000\000\000", !19, !2, null, !8, null, null, null} ; [ DW_TAG_structure_type ] [Rect] [line 6, size 256, align 64, offset 0] [def] [from ]
!8 = !{!9, !14}
!9 = !{!"0xd\00P1\007\00128\0064\000\000", !19, !7, !10} ; [ DW_TAG_member ]
!10 = !{!"0x13\00Pt\001\00128\0064\000\000\000", !19, !2, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [Pt] [line 1, size 128, align 64, offset 0] [def] [from ]
!11 = !{!12, !13}
!12 = !{!"0xd\00x\002\0064\0064\000\000", !19, !10, !6} ; [ DW_TAG_member ]
!13 = !{!"0xd\00y\003\0064\0064\0064\000", !19, !10, !6} ; [ DW_TAG_member ]
!14 = !{!"0xd\00P2\008\00128\0064\00128\000", !19, !7, !10} ; [ DW_TAG_member ]
!15 = !MDLocation(line: 11, scope: !1)
!16 = !MDLocation(line: 12, scope: !17)
!17 = !{!"0xb\0011\000\000", !19, !1} ; [ DW_TAG_lexical_block ]
!18 = !{!1}
!19 = !{!"b2.c", !"/tmp/"}
!20 = !{i32 0}
!21 = !{i32 1, !"Debug Info Version", i32 2}
