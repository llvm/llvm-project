; RUN: llc -O0 -mtriple=arm-apple-darwin < %s | grep DW_OP_breg
; Use DW_OP_breg in variable's location expression if the variable is in a stack slot.

%struct.SVal = type { i8*, i32 }

define i32 @_Z3fooi4SVal(i32 %i, %struct.SVal* noalias %location) nounwind ssp {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !23, metadata !{!"0x102"}), !dbg !24
  call void @llvm.dbg.value(metadata %struct.SVal* %location, i64 0, metadata !25, metadata !{!"0x102"}), !dbg !24
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

define linkonce_odr void @_ZN4SValC1Ev(%struct.SVal* %this) nounwind ssp align 2  {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.value(metadata %struct.SVal* %this, i64 0, metadata !31, metadata !{!"0x102"}), !dbg !34
  %0 = getelementptr inbounds %struct.SVal* %this, i32 0, i32 0, !dbg !34 ; <i8**> [#uses=1]
  store i8* null, i8** %0, align 8, !dbg !34
  %1 = getelementptr inbounds %struct.SVal* %this, i32 0, i32 1, !dbg !34 ; <i32*> [#uses=1]
  store i32 0, i32* %1, align 8, !dbg !34
  br label %return, !dbg !34

return:                                           ; preds = %entry
  ret void, !dbg !35
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp {
entry:
  %0 = alloca %struct.SVal                        ; <%struct.SVal*> [#uses=3]
  %v = alloca %struct.SVal                        ; <%struct.SVal*> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.SVal* %v, metadata !38, metadata !{!"0x102"}), !dbg !41
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
  call void @llvm.dbg.value(metadata i32 %8, i64 0, metadata !44, metadata !{!"0x102"}), !dbg !43
  br label %return, !dbg !45

return:                                           ; preds = %entry
  ret i32 0, !dbg !45
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!49}

!0 = !{!"0x2e\00SVal\00SVal\00\0011\000\000\000\006\000\000\000", !48, !1, !14, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x13\00SVal\001\00128\0064\000\000\000", !48, null, null, !4, null, null, null} ; [ DW_TAG_structure_type ] [SVal] [line 1, size 128, align 64, offset 0] [def] [from ]
!2 = !{!"0x29", !48} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\004\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\001", !48, !47, !47, !46, !47,  !47} ; [ DW_TAG_compile_unit ]
!4 = !{!5, !7, !0, !9}
!5 = !{!"0xd\00Data\007\0064\0064\000\000", !48, !1, !6} ; [ DW_TAG_member ]
!6 = !{!"0xf\00\000\0064\0064\000\000", !48, null, null} ; [ DW_TAG_pointer_type ]
!7 = !{!"0xd\00Kind\008\0032\0032\0064\000", !48, !1, !8} ; [ DW_TAG_member ]
!8 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", !48, null} ; [ DW_TAG_base_type ]
!9 = !{!"0x2e\00~SVal\00~SVal\00\0012\000\000\000\006\000\000\000", !48, !1, !10, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!10 = !{!"0x15\00\000\000\000\000\000\000", !48, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null, !12, !13}
!12 = !{!"0xf\00\000\0064\0064\000\0064", !48, null, !1} ; [ DW_TAG_pointer_type ]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", !48, null} ; [ DW_TAG_base_type ]
!14 = !{!"0x15\00\000\000\000\000\000\000", !48, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null, !12}
!16 = !{!"0x2e\00SVal\00SVal\00_ZN4SValC1Ev\0011\000\001\000\006\000\000\000", !48, !1, !14, null, void (%struct.SVal*)* @_ZN4SValC1Ev, null, null, null} ; [ DW_TAG_subprogram ]
!17 = !{!"0x2e\00foo\00foo\00_Z3fooi4SVal\0016\000\001\000\006\000\000\000", !48, !2, !18, null, i32 (i32, %struct.SVal*)* @_Z3fooi4SVal, null, null, null} ; [ DW_TAG_subprogram ]
!18 = !{!"0x15\00\000\000\000\000\000\000", !48, null, null, !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = !{!13, !13, !1}
!20 = !{!"0x2e\00main\00main\00main\0023\000\001\000\006\000\000\000", !48, !2, !21, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!21 = !{!"0x15\00\000\000\000\000\000\000", !48, null, null, !22, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = !{!13}
!23 = !{!"0x101\00i\0016\000", !17, !2, !13} ; [ DW_TAG_arg_variable ]
!24 = !MDLocation(line: 16, scope: !17)
!25 = !{!"0x101\00location\0016\000", !17, !2, !26} ; [ DW_TAG_arg_variable ]
!26 = !{!"0x10\00SVal\000\0064\0064\000\000", !48, !2, !1} ; [ DW_TAG_reference_type ]
!27 = !MDLocation(line: 17, scope: !28)
!28 = !{!"0xb\0016\000\002", !2, !17} ; [ DW_TAG_lexical_block ]
!29 = !MDLocation(line: 18, scope: !28)
!30 = !MDLocation(line: 20, scope: !28)
!31 = !{!"0x101\00this\0011\000", !16, !2, !32} ; [ DW_TAG_arg_variable ]
!32 = !{!"0x26\00\000\0064\0064\000\0064", !48, !2, !33} ; [ DW_TAG_const_type ]
!33 = !{!"0xf\00\000\0064\0064\000\000", !48, !2, !1} ; [ DW_TAG_pointer_type ]
!34 = !MDLocation(line: 11, scope: !16)
!35 = !MDLocation(line: 11, scope: !36)
!36 = !{!"0xb\0011\000\001", !48, !37} ; [ DW_TAG_lexical_block ]
!37 = !{!"0xb\0011\000\000", !48, !16} ; [ DW_TAG_lexical_block ]
!38 = !{!"0x100\00v\0024\000", !39, !2, !1} ; [ DW_TAG_auto_variable ]
!39 = !{!"0xb\0023\000\004", !48, !40} ; [ DW_TAG_lexical_block ]
!40 = !{!"0xb\0023\000\003", !48, !20} ; [ DW_TAG_lexical_block ]
!41 = !MDLocation(line: 24, scope: !39)
!42 = !MDLocation(line: 25, scope: !39)
!43 = !MDLocation(line: 26, scope: !39)
!44 = !{!"0x100\00k\0026\000", !39, !2, !13} ; [ DW_TAG_auto_variable ]
!45 = !MDLocation(line: 27, scope: !39)
!46 = !{!16, !17, !20}
!47 = !{}
!48 = !{!"small.cc", !"/Users/manav/R8248330"}
!49 = !{i32 1, !"Debug Info Version", i32 2}
