; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s | grep DW_OP_breg7
; Use DW_OP_breg7 in variable's location expression if the variable is in a stack slot.

%struct.SVal = type { i8*, i32 }

define i32 @_Z3fooi4SVal(i32 %i, %struct.SVal* noalias %location) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !23), !dbg !24
  call void @llvm.dbg.value(metadata !{%struct.SVal* %location}, i64 0, metadata !25), !dbg !24
  %0 = icmp ne i32 %i, 0, !dbg !27                ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1, !dbg !27

bb:                                               ; preds = %entry
  %1 = getelementptr inbounds %struct.SVal* %location, i32 0, i32 1, !dbg !29 ; <i32*> [#uses=1]
  %2 = load i32* %1, align 8, !dbg !29            ; <i32> [#uses=1]
  %3 = add i32 %2, %i, !dbg !29                   ; <i32> [#uses=1]
  br label %bb2, !dbg !29

bb1:                                              ; preds = %entry
  %4 = getelementptr inbounds %struct.SVal* %location, i32 0, i32 1, !dbg !30 ; <i32*> [#uses=1]
  %5 = load i32* %4, align 8, !dbg !30            ; <i32> [#uses=1]
  %6 = sub i32 %5, 1, !dbg !30                    ; <i32> [#uses=1]
  br label %bb2, !dbg !30

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ %3, %bb ], [ %6, %bb1 ]         ; <i32> [#uses=1]
  br label %return, !dbg !29

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !29
}

define linkonce_odr void @_ZN4SValC1Ev(%struct.SVal* %this) nounwind ssp align 2 {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata !{%struct.SVal* %this}, i64 0, metadata !31), !dbg !34
  %0 = getelementptr inbounds %struct.SVal* %this, i32 0, i32 0, !dbg !34 ; <i8**> [#uses=1]
  store i8* null, i8** %0, align 8, !dbg !34
  %1 = getelementptr inbounds %struct.SVal* %this, i32 0, i32 1, !dbg !34 ; <i32*> [#uses=1]
  store i32 0, i32* %1, align 8, !dbg !34
  br label %return, !dbg !34

return:                                           ; preds = %entry
  ret void, !dbg !35
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp {
entry:
  %0 = alloca %struct.SVal                        ; <%struct.SVal*> [#uses=3]
  %v = alloca %struct.SVal                        ; <%struct.SVal*> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.SVal* %v}, metadata !38), !dbg !41
  call void @_ZN4SValC1Ev(%struct.SVal* %v) nounwind, !dbg !41
  %1 = getelementptr inbounds %struct.SVal* %v, i32 0, i32 1, !dbg !42 ; <i32*> [#uses=1]
  store i32 1, i32* %1, align 8, !dbg !42
  %2 = getelementptr inbounds %struct.SVal* %0, i32 0, i32 0, !dbg !43 ; <i8**> [#uses=1]
  %3 = getelementptr inbounds %struct.SVal* %v, i32 0, i32 0, !dbg !43 ; <i8**> [#uses=1]
  %4 = load i8** %3, align 8, !dbg !43            ; <i8*> [#uses=1]
  store i8* %4, i8** %2, align 8, !dbg !43
  %5 = getelementptr inbounds %struct.SVal* %0, i32 0, i32 1, !dbg !43 ; <i32*> [#uses=1]
  %6 = getelementptr inbounds %struct.SVal* %v, i32 0, i32 1, !dbg !43 ; <i32*> [#uses=1]
  %7 = load i32* %6, align 8, !dbg !43            ; <i32> [#uses=1]
  store i32 %7, i32* %5, align 8, !dbg !43
  %8 = call i32 @_Z3fooi4SVal(i32 2, %struct.SVal* noalias %0) nounwind, !dbg !43 ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata !{i32 %8}, i64 0, metadata !44), !dbg !43
  br label %return, !dbg !45

return:                                           ; preds = %entry
  ret i32 0, !dbg !45
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0, !9, !16, !17, !20}

!0 = metadata !{i32 524334, i32 0, metadata !1, metadata !"SVal", metadata !"SVal", metadata !"", metadata !2, i32 11, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 524307, metadata !2, metadata !"SVal", metadata !2, i32 1, i64 128, i64 64, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_structure_type ]
!2 = metadata !{i32 524329, metadata !"small.cc", metadata !"/Users/manav/R8248330", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 4, metadata !"small.cc", metadata !"/Users/manav/R8248330", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !5, metadata !7, metadata !0, metadata !9}
!5 = metadata !{i32 524301, metadata !1, metadata !"Data", metadata !2, i32 7, i64 64, i64 64, i64 0, i32 0, metadata !6} ; [ DW_TAG_member ]
!6 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 524301, metadata !1, metadata !"Kind", metadata !2, i32 8, i64 32, i64 32, i64 64, i32 0, metadata !8} ; [ DW_TAG_member ]
!8 = metadata !{i32 524324, metadata !2, metadata !"unsigned int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524334, i32 0, metadata !1, metadata !"~SVal", metadata !"~SVal", metadata !"", metadata !2, i32 12, metadata !10, i1 false, i1 false, i32 0, i32 0, null, i1 false, i1 false, null} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_subroutine_type ]
!11 = metadata !{null, metadata !12, metadata !13}
!12 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !1} ; [ DW_TAG_pointer_type ]
!13 = metadata !{i32 524324, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null} ; [ DW_TAG_subroutine_type ]
!15 = metadata !{null, metadata !12}
!16 = metadata !{i32 524334, i32 0, metadata !1, metadata !"SVal", metadata !"SVal", metadata !"_ZN4SValC1Ev", metadata !2, i32 11, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, void (%struct.SVal*)* @_ZN4SValC1Ev} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 524334, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"_Z3fooi4SVal", metadata !2, i32 16, metadata !18, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 (i32, %struct.SVal*)* @_Z3fooi4SVal} ; [ DW_TAG_subprogram ]
!18 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !19, i32 0, null} ; [ DW_TAG_subroutine_type ]
!19 = metadata !{metadata !13, metadata !13, metadata !1}
!20 = metadata !{i32 524334, i32 0, metadata !2, metadata !"main", metadata !"main", metadata !"main", metadata !2, i32 23, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!21 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !22, i32 0, null} ; [ DW_TAG_subroutine_type ]
!22 = metadata !{metadata !13}
!23 = metadata !{i32 524545, metadata !17, metadata !"i", metadata !2, i32 16, metadata !13} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 16, i32 0, metadata !17, null}
!25 = metadata !{i32 524545, metadata !17, metadata !"location", metadata !2, i32 16, metadata !26} ; [ DW_TAG_arg_variable ]
!26 = metadata !{i32 524304, metadata !2, metadata !"SVal", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !1} ; [ DW_TAG_reference_type ]
!27 = metadata !{i32 17, i32 0, metadata !28, null}
!28 = metadata !{i32 524299, metadata !17, i32 16, i32 0, metadata !2, i32 2} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 18, i32 0, metadata !28, null}
!30 = metadata !{i32 20, i32 0, metadata !28, null}
!31 = metadata !{i32 524545, metadata !16, metadata !"this", metadata !2, i32 11, metadata !32} ; [ DW_TAG_arg_variable ]
!32 = metadata !{i32 524326, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !33} ; [ DW_TAG_const_type ]
!33 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !1} ; [ DW_TAG_pointer_type ]
!34 = metadata !{i32 11, i32 0, metadata !16, null}
!35 = metadata !{i32 11, i32 0, metadata !36, null}
!36 = metadata !{i32 524299, metadata !37, i32 11, i32 0, metadata !2, i32 1} ; [ DW_TAG_lexical_block ]
!37 = metadata !{i32 524299, metadata !16, i32 11, i32 0, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!38 = metadata !{i32 524544, metadata !39, metadata !"v", metadata !2, i32 24, metadata !1} ; [ DW_TAG_auto_variable ]
!39 = metadata !{i32 524299, metadata !40, i32 23, i32 0, metadata !2, i32 4} ; [ DW_TAG_lexical_block ]
!40 = metadata !{i32 524299, metadata !20, i32 23, i32 0, metadata !2, i32 3} ; [ DW_TAG_lexical_block ]
!41 = metadata !{i32 24, i32 0, metadata !39, null}
!42 = metadata !{i32 25, i32 0, metadata !39, null}
!43 = metadata !{i32 26, i32 0, metadata !39, null}
!44 = metadata !{i32 524544, metadata !39, metadata !"k", metadata !2, i32 26, metadata !13} ; [ DW_TAG_auto_variable ]
!45 = metadata !{i32 27, i32 0, metadata !39, null}
