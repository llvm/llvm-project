; RUN: llc -march=x86 -O0 < %s | FileCheck %s
; Currently, dbg.declare generates a DEBUG_VALUE comment.  Eventually it will
; generate DWARF and this test will need to be modified or removed.


%struct.Pt = type { double, double }
%struct.Rect = type { %struct.Pt, %struct.Pt }

define double @foo(%struct.Rect* byval %my_r0) nounwind ssp {
entry:
;CHECK: DEBUG_VALUE
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.Rect* %my_r0}, metadata !0), !dbg !15
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

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!21}

!0 = metadata !{i32 786689, metadata !1, metadata !"my_r0", metadata !2, i32 11, metadata !7, i32 0, null} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 786478, metadata !19, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", i32 11, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, double (%struct.Rect*)* @foo, null, null, null, i32 11} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !19} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786449, metadata !19, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 false, metadata !"", i32 0, metadata !20, metadata !20, metadata !18, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 786453, metadata !19, metadata !2, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !7}
!6 = metadata !{i32 786468, metadata !19, metadata !2, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 786451, metadata !19, metadata !2, metadata !"Rect", i32 6, i64 256, i64 64, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [Rect] [line 6, size 256, align 64, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !14}
!9 = metadata !{i32 786445, metadata !19, metadata !7, metadata !"P1", i32 7, i64 128, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 786451, metadata !19, metadata !2, metadata !"Pt", i32 1, i64 128, i64 64, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [Pt] [line 1, size 128, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{i32 786445, metadata !19, metadata !10, metadata !"x", i32 2, i64 64, i64 64, i64 0, i32 0, metadata !6} ; [ DW_TAG_member ]
!13 = metadata !{i32 786445, metadata !19, metadata !10, metadata !"y", i32 3, i64 64, i64 64, i64 64, i32 0, metadata !6} ; [ DW_TAG_member ]
!14 = metadata !{i32 786445, metadata !19, metadata !7, metadata !"P2", i32 8, i64 128, i64 64, i64 128, i32 0, metadata !10} ; [ DW_TAG_member ]
!15 = metadata !{i32 11, i32 0, metadata !1, null}
!16 = metadata !{i32 12, i32 0, metadata !17, null}
!17 = metadata !{i32 786443, metadata !19, metadata !1, i32 11, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!18 = metadata !{metadata !1}
!19 = metadata !{metadata !"b2.c", metadata !"/tmp/"}
!20 = metadata !{i32 0}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
