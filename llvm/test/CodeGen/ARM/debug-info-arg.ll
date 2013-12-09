; RUN: llc < %s | FileCheck %s
; Test to check argument y's debug info uses FI
; Radar 10048772
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.tag_s = type { i32, i32, i32 }

define void @foo(%struct.tag_s* nocapture %this, %struct.tag_s* %c, i64 %x, i64 %y, %struct.tag_s* nocapture %ptr1, %struct.tag_s* nocapture %ptr2) nounwind ssp {
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %this}, i64 0, metadata !5), !dbg !20
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %c}, i64 0, metadata !13), !dbg !21
  tail call void @llvm.dbg.value(metadata !{i64 %x}, i64 0, metadata !14), !dbg !22
  tail call void @llvm.dbg.value(metadata !{i64 %y}, i64 0, metadata !17), !dbg !23
;CHECK:	@DEBUG_VALUE: foo:y <- [R7+8]
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %ptr1}, i64 0, metadata !18), !dbg !24
  tail call void @llvm.dbg.value(metadata !{%struct.tag_s* %ptr2}, i64 0, metadata !19), !dbg !25
  %1 = icmp eq %struct.tag_s* %c, null, !dbg !26
  br i1 %1, label %3, label %2, !dbg !26

; <label>:2                                       ; preds = %0
  tail call void @foobar(i64 %x, i64 %y) nounwind, !dbg !28
  br label %3, !dbg !28

; <label>:3                                       ; preds = %0, %2
  ret void, !dbg !29
}

declare void @foobar(i64, i64)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = metadata !{i32 786449, metadata !32, i32 12, metadata !"Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)", i1 true, metadata !"", i32 0, metadata !4, metadata !4, metadata !30, null,  null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 786478, metadata !2, metadata !2, metadata !"foo", metadata !"foo", metadata !"", i32 11, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%struct.tag_s*, %struct.tag_s*, i64, i64, %struct.tag_s*, %struct.tag_s*)* @foo, null, null, metadata !31, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [foo]
!2 = metadata !{i32 786473, metadata !32} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786453, metadata !32, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{i32 786689, metadata !1, metadata !"this", metadata !2, i32 16777227, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!6 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 786451, metadata !32, metadata !0, metadata !"tag_s", i32 5, i64 96, i64 32, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [tag_s] [line 5, size 96, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !11, metadata !12}
!9 = metadata !{i32 786445, metadata !32, metadata !7, metadata !"x", i32 6, i64 32, i64 32, i64 0, i32 0, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 786468, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 786445, metadata !32, metadata !7, metadata !"y", i32 7, i64 32, i64 32, i64 32, i32 0, metadata !10} ; [ DW_TAG_member ]
!12 = metadata !{i32 786445, metadata !32, metadata !7, metadata !"z", i32 8, i64 32, i64 32, i64 64, i32 0, metadata !10} ; [ DW_TAG_member ]
!13 = metadata !{i32 786689, metadata !1, metadata !"c", metadata !2, i32 33554443, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!14 = metadata !{i32 786689, metadata !1, metadata !"x", metadata !2, i32 50331659, metadata !15, i32 0, null} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 786454, metadata !32, metadata !0, metadata !"UInt64", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ]
!16 = metadata !{i32 786468, null, metadata !0, metadata !"long long unsigned int", i32 0, i64 64, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 786689, metadata !1, metadata !"y", metadata !2, i32 67108875, metadata !15, i32 0, null} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 786689, metadata !1, metadata !"ptr1", metadata !2, i32 83886091, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 786689, metadata !1, metadata !"ptr2", metadata !2, i32 100663307, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 11, i32 24, metadata !1, null}
!21 = metadata !{i32 11, i32 44, metadata !1, null}
!22 = metadata !{i32 11, i32 54, metadata !1, null}
!23 = metadata !{i32 11, i32 64, metadata !1, null}
!24 = metadata !{i32 11, i32 81, metadata !1, null}
!25 = metadata !{i32 11, i32 101, metadata !1, null}
!26 = metadata !{i32 12, i32 3, metadata !27, null}
!27 = metadata !{i32 786443, metadata !1, i32 11, i32 107, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 13, i32 5, metadata !27, null}
!29 = metadata !{i32 14, i32 1, metadata !27, null}
!30 = metadata !{metadata !1}
!31 = metadata !{metadata !5, metadata !13, metadata !14, metadata !17, metadata !18, metadata!19}
!32 = metadata !{metadata !"one.c", metadata !"/Volumes/Athwagate/R10048772"}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
