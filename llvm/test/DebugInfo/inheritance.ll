; RUN: llc %s -o /dev/null
; PR 2613.

%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { i8*, i8* }
%struct.test1 = type { i32 (...)** }

@_ZTV5test1 = weak_odr constant [4 x i32 (...)*] [i32 (...)* null, i32 (...)* bitcast (%struct.__class_type_info_pseudo* @_ZTI5test1 to i32 (...)*), i32 (...)* bitcast (void (%struct.test1*)* @_ZN5test1D1Ev to i32 (...)*), i32 (...)* bitcast (void (%struct.test1*)* @_ZN5test1D0Ev to i32 (...)*)], align 32 ; <[4 x i32 (...)*]*> [#uses=1]
@_ZTI5test1 = weak_odr constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i64 add (i64 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i64), i64 16) to i8*), i8* getelementptr inbounds ([7 x i8]* @_ZTS5test1, i64 0, i64 0) } }, align 16 ; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external constant [0 x i32 (...)*] ; <[0 x i32 (...)*]*> [#uses=1]
@_ZTS5test1 = weak_odr constant [7 x i8] c"5test1\00" ; <[7 x i8]*> [#uses=2]

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %tst = alloca %struct.test1                     ; <%struct.test1*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.test1* %tst}, metadata !0), !dbg !21
  call void @_ZN5test1C1Ev(%struct.test1* %tst) nounwind, !dbg !22
  store i32 0, i32* %0, align 4, !dbg !23
  %1 = load i32* %0, align 4, !dbg !23            ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !23
  br label %return, !dbg !23

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !23          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !23
}

define linkonce_odr void @_ZN5test1C1Ev(%struct.test1* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.test1*              ; <%struct.test1**> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.test1** %this_addr}, metadata !24), !dbg !28
  store %struct.test1* %this, %struct.test1** %this_addr
  %0 = load %struct.test1** %this_addr, align 8, !dbg !28 ; <%struct.test1*> [#uses=1]
  %1 = getelementptr inbounds %struct.test1* %0, i32 0, i32 0, !dbg !28 ; <i32 (...)***> [#uses=1]
  store i32 (...)** getelementptr inbounds ([4 x i32 (...)*]* @_ZTV5test1, i64 0, i64 2), i32 (...)*** %1, align 8, !dbg !28
  br label %return, !dbg !28

return:                                           ; preds = %entry
  ret void, !dbg !29
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN5test1D1Ev(%struct.test1* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.test1*              ; <%struct.test1**> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.test1** %this_addr}, metadata !32), !dbg !34
  store %struct.test1* %this, %struct.test1** %this_addr
  %0 = load %struct.test1** %this_addr, align 8, !dbg !35 ; <%struct.test1*> [#uses=1]
  %1 = getelementptr inbounds %struct.test1* %0, i32 0, i32 0, !dbg !35 ; <i32 (...)***> [#uses=1]
  store i32 (...)** getelementptr inbounds ([4 x i32 (...)*]* @_ZTV5test1, i64 0, i64 2), i32 (...)*** %1, align 8, !dbg !35
  br label %bb, !dbg !37

bb:                                               ; preds = %entry
  %2 = trunc i32 0 to i8, !dbg !37                ; <i8> [#uses=1]
  %toBool = icmp ne i8 %2, 0, !dbg !37            ; <i1> [#uses=1]
  br i1 %toBool, label %bb1, label %bb2, !dbg !37

bb1:                                              ; preds = %bb
  %3 = load %struct.test1** %this_addr, align 8, !dbg !37 ; <%struct.test1*> [#uses=1]
  %4 = bitcast %struct.test1* %3 to i8*, !dbg !37 ; <i8*> [#uses=1]
  call void @_ZdlPv(i8* %4) nounwind, !dbg !37
  br label %bb2, !dbg !37

bb2:                                              ; preds = %bb1, %bb
  br label %return, !dbg !37

return:                                           ; preds = %bb2
  ret void, !dbg !37
}

define linkonce_odr void @_ZN5test1D0Ev(%struct.test1* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.test1*              ; <%struct.test1**> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.test1** %this_addr}, metadata !38), !dbg !40
  store %struct.test1* %this, %struct.test1** %this_addr
  %0 = load %struct.test1** %this_addr, align 8, !dbg !41 ; <%struct.test1*> [#uses=1]
  %1 = getelementptr inbounds %struct.test1* %0, i32 0, i32 0, !dbg !41 ; <i32 (...)***> [#uses=1]
  store i32 (...)** getelementptr inbounds ([4 x i32 (...)*]* @_ZTV5test1, i64 0, i64 2), i32 (...)*** %1, align 8, !dbg !41
  br label %bb, !dbg !43

bb:                                               ; preds = %entry
  %2 = trunc i32 1 to i8, !dbg !43                ; <i8> [#uses=1]
  %toBool = icmp ne i8 %2, 0, !dbg !43            ; <i1> [#uses=1]
  br i1 %toBool, label %bb1, label %bb2, !dbg !43

bb1:                                              ; preds = %bb
  %3 = load %struct.test1** %this_addr, align 8, !dbg !43 ; <%struct.test1*> [#uses=1]
  %4 = bitcast %struct.test1* %3 to i8*, !dbg !43 ; <i8*> [#uses=1]
  call void @_ZdlPv(i8* %4) nounwind, !dbg !43
  br label %bb2, !dbg !43

bb2:                                              ; preds = %bb1, %bb
  br label %return, !dbg !43

return:                                           ; preds = %bb2
  ret void, !dbg !43
}

declare void @_ZdlPv(i8*) nounwind

!0 = metadata !{i32 459008, metadata !1, metadata !"tst", metadata !4, i32 13, metadata !8} ; [ DW_TAG_auto_variable ]
!1 = metadata !{i32 458763, metadata !2, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 458763, metadata !3, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 458798, i32 0, metadata !4, metadata !"main", metadata !"main", metadata !"main", metadata !4, i32 11, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 458769, i32 0, i32 4, metadata !"inheritance.cpp", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 458773, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null} ; [ DW_TAG_subroutine_type ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 458788, metadata !4, metadata !"int", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 458771, metadata !4, metadata !"test1", metadata !4, i32 1, i64 64, i64 64, i64 0, i32 0, null, metadata !9, i32 0, metadata !8} ; [ DW_TAG_structure_type ]
!9 = metadata !{metadata !10, metadata !14, metadata !18}
!10 = metadata !{i32 458765, metadata !8, metadata !"_vptr$test1", metadata !4, i32 1, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_member ]
!11 = metadata !{i32 458767, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 458767, metadata !4, metadata !"__vtbl_ptr_type", metadata !13, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !5} ; [ DW_TAG_pointer_type ]
!13 = metadata !{i32 458769, i32 0, i32 4, metadata !"<built-in>", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 false, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!14 = metadata !{i32 458798, i32 0, metadata !8, metadata !"test1", metadata !"test1", metadata !"", metadata !4, i32 1, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i1 true} ; [ DW_TAG_subprogram ]
!15 = metadata !{i32 458773, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null} ; [ DW_TAG_subroutine_type ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{i32 458767, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !8} ; [ DW_TAG_pointer_type ]
!18 = metadata !{i32 458798, i32 0, metadata !8, metadata !"~test1", metadata !"~test1", metadata !"", metadata !4, i32 4, metadata !19, i1 false, i1 false, i32 1, i32 0, metadata !8, i1 false} ; [ DW_TAG_subprogram ]
!19 = metadata !{i32 458773, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !20, i32 0, null} ; [ DW_TAG_subroutine_type ]
!20 = metadata !{null, metadata !17, metadata !7}
!21 = metadata !{i32 11, i32 0, metadata !1, null}
!22 = metadata !{i32 13, i32 0, metadata !1, null}
!23 = metadata !{i32 14, i32 0, metadata !1, null}
!24 = metadata !{i32 459009, metadata !25, metadata !"this", metadata !4, i32 13, metadata !26} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 458798, i32 0, metadata !4, metadata !"test1", metadata !"test1", metadata !"_ZN5test1C1Ev", metadata !4, i32 1, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!26 = metadata !{i32 458790, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !27} ; [ DW_TAG_const_type ]
!27 = metadata !{i32 458767, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!28 = metadata !{i32 1, i32 0, metadata !25, null}
!29 = metadata !{i32 1, i32 0, metadata !30, null}
!30 = metadata !{i32 458763, metadata !31, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!31 = metadata !{i32 458763, metadata !25, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!32 = metadata !{i32 459009, metadata !33, metadata !"this", metadata !4, i32 4, metadata !26} ; [ DW_TAG_arg_variable ]
!33 = metadata !{i32 458798, i32 0, metadata !8, metadata !"~test1", metadata !"~test1", metadata !"_ZN5test1D1Ev", metadata !4, i32 4, metadata !15, i1 false, i1 true, i32 1, i32 0, metadata !8, i1 false} ; [ DW_TAG_subprogram ]
!34 = metadata !{i32 4, i32 0, metadata !33, null}
!35 = metadata !{i32 5, i32 0, metadata !36, null}
!36 = metadata !{i32 458763, metadata !33, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!37 = metadata !{i32 6, i32 0, metadata !36, null}
!38 = metadata !{i32 459009, metadata !39, metadata !"this", metadata !4, i32 4, metadata !26} ; [ DW_TAG_arg_variable ]
!39 = metadata !{i32 458798, i32 0, metadata !8, metadata !"~test1", metadata !"~test1", metadata !"_ZN5test1D0Ev", metadata !4, i32 4, metadata !15, i1 false, i1 true, i32 1, i32 1, metadata !8, i1 false} ; [ DW_TAG_subprogram ]
!40 = metadata !{i32 4, i32 0, metadata !39, null}
!41 = metadata !{i32 5, i32 0, metadata !42, null}
!42 = metadata !{i32 458763, metadata !39, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!43 = metadata !{i32 6, i32 0, metadata !42, null}
