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
  call void @llvm.dbg.declare(metadata %struct.test1* %tst, metadata !0, metadata !{!"0x102"}), !dbg !21
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
  call void @llvm.dbg.declare(metadata %struct.test1** %this_addr, metadata !24, metadata !{!"0x102"}), !dbg !28
  store %struct.test1* %this, %struct.test1** %this_addr
  %0 = load %struct.test1** %this_addr, align 8, !dbg !28 ; <%struct.test1*> [#uses=1]
  %1 = getelementptr inbounds %struct.test1* %0, i32 0, i32 0, !dbg !28 ; <i32 (...)***> [#uses=1]
  store i32 (...)** getelementptr inbounds ([4 x i32 (...)*]* @_ZTV5test1, i64 0, i64 2), i32 (...)*** %1, align 8, !dbg !28
  br label %return, !dbg !28

return:                                           ; preds = %entry
  ret void, !dbg !29
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN5test1D1Ev(%struct.test1* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.test1*              ; <%struct.test1**> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.test1** %this_addr, metadata !32, metadata !{!"0x102"}), !dbg !34
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
  call void @llvm.dbg.declare(metadata %struct.test1** %this_addr, metadata !38, metadata !{!"0x102"}), !dbg !40
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

!0 = !{!"0x100\00tst\0013\000", !1, !4, !8} ; [ DW_TAG_auto_variable ]
!1 = !{!"0xb\000\000\000", !44, !2} ; [ DW_TAG_lexical_block ]
!2 = !{!"0xb\000\000\000", !44, !3} ; [ DW_TAG_lexical_block ]
!3 = !{!"0x2e\00main\00main\00main\0011\000\001\000\006\000\000\000", i32 0, !4, !5, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!4 = !{!"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !44, !45, !45, null, null, null} ; [ DW_TAG_compile_unit ]
!5 = !{!"0x15\00\000\000\000\000\000\000", !4, null, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = !{!7}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !4} ; [ DW_TAG_base_type ]
!8 = !{!"0x13\00test1\001\0064\0064\000\000\000", !44, !4, null, !9, !8, null, null} ; [ DW_TAG_structure_type ] [test1] [line 1, size 64, align 64, offset 0] [def] [from ]
!9 = !{!10, !14, !18}
!10 = !{!"0xd\00_vptr$test1\001\0064\0064\000\000", !44, !8, !11} ; [ DW_TAG_member ]
!11 = !{!"0xf\00\000\0064\0064\000\000", !4, null, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0xf\00__vtbl_ptr_type\000\000\000\000\000", null, !4, !5} ; [ DW_TAG_pointer_type ]
!13 = !{!"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !46, !45, !45, null, null, null} ; [ DW_TAG_compile_unit ]
!14 = !{!"0x2e\00test1\00test1\00\001\000\000\000\006\001\000\000", i32 0, !8, !15, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!15 = !{!"0x15\00\000\000\000\000\000\000", !4, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !17}
!17 = !{!"0xf\00\000\0064\0064\000\0064", !4, null, !8} ; [ DW_TAG_pointer_type ]
!18 = !{!"0x2e\00~test1\00~test1\00\004\000\000\001\006\000\000\000", i32 0, !8, !19, !8, null, null, null, null} ; [ DW_TAG_subprogram ]
!19 = !{!"0x15\00\000\000\000\000\000\000", !4, null, null, !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = !{null, !17, !7}
!21 = !MDLocation(line: 11, scope: !1)
!22 = !MDLocation(line: 13, scope: !1)
!23 = !MDLocation(line: 14, scope: !1)
!24 = !{!"0x101\00this\0013\000", !25, !4, !26} ; [ DW_TAG_arg_variable ]
!25 = !{!"0x2e\00test1\00test1\00_ZN5test1C1Ev\001\000\001\000\006\000\000\000", i32 0, !4, !15, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!26 = !{!"0x26\00\000\0064\0064\000\0064", !4, null, !27} ; [ DW_TAG_const_type ]
!27 = !{!"0xf\00\000\0064\0064\000\000", !4, null, !8} ; [ DW_TAG_pointer_type ]
!28 = !MDLocation(line: 1, scope: !25)
!29 = !MDLocation(line: 1, scope: !30)
!30 = !{!"0xb\000\000\000", !44, !31} ; [ DW_TAG_lexical_block ]
!31 = !{!"0xb\000\000\000", !44, !25} ; [ DW_TAG_lexical_block ]
!32 = !{!"0x101\00this\004\000", !33, !4, !26} ; [ DW_TAG_arg_variable ]
!33 = !{!"0x2e\00~test1\00~test1\00_ZN5test1D1Ev\004\000\001\001\006\000\000\000", i32 0, !8, !15, !8, null, null, null, null} ; [ DW_TAG_subprogram ]
!34 = !MDLocation(line: 4, scope: !33)
!35 = !MDLocation(line: 5, scope: !36)
!36 = !{!"0xb\000\000\000", !44, !33} ; [ DW_TAG_lexical_block ]
!37 = !MDLocation(line: 6, scope: !36)
!38 = !{!"0x101\00this\004\000", !39, !4, !26} ; [ DW_TAG_arg_variable ]
!39 = !{!"0x2e\00~test1\00~test1\00_ZN5test1D0Ev\004\000\001\001\006\000\000\000", i32 0, !8, !15, !8, null, null, null, null} ; [ DW_TAG_subprogram ]
!40 = !MDLocation(line: 4, scope: !39)
!41 = !MDLocation(line: 5, scope: !42)
!42 = !{!"0xb\000\000\000", !44, !39} ; [ DW_TAG_lexical_block ]
!43 = !MDLocation(line: 6, scope: !42)
!44 = !{!"inheritance.cpp", !"/tmp/"}
!45 = !{i32 0}
!46 = !{!"<built-in>", !"/tmp/"}
