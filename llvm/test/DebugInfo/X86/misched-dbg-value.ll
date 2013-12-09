; RUN: llc %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -enable-misched
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13183203
; Make sure when misched is enabled, we still have location information for
; function parameters.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_subprogram
; CHECK: Proc8
; CHECK: DW_TAG_formal_parameter
; CHECK: Array1Par
; CHECK: DW_AT_location
; CHECK: DW_TAG_formal_parameter
; CHECK: Array2Par
; CHECK: DW_AT_location
; CHECK: DW_TAG_formal_parameter
; CHECK: IntParI1
; CHECK: DW_AT_location
; CHECK: DW_TAG_formal_parameter
; CHECK: IntParI2
; CHECK: DW_AT_location

%struct.Record = type { %struct.Record*, i32, i32, i32, [31 x i8] }

@Version = global [4 x i8] c"1.1\00", align 1
@IntGlob = common global i32 0, align 4
@BoolGlob = common global i32 0, align 4
@Char1Glob = common global i8 0, align 1
@Char2Glob = common global i8 0, align 1
@Array1Glob = common global [51 x i32] zeroinitializer, align 16
@Array2Glob = common global [51 x [51 x i32]] zeroinitializer, align 16
@PtrGlb = common global %struct.Record* null, align 8
@PtrGlbNext = common global %struct.Record* null, align 8

define void @Proc8(i32* nocapture %Array1Par, [51 x i32]* nocapture %Array2Par, i32 %IntParI1, i32 %IntParI2) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i32* %Array1Par}, i64 0, metadata !23), !dbg !64
  tail call void @llvm.dbg.value(metadata !{[51 x i32]* %Array2Par}, i64 0, metadata !24), !dbg !65
  tail call void @llvm.dbg.value(metadata !{i32 %IntParI1}, i64 0, metadata !25), !dbg !66
  tail call void @llvm.dbg.value(metadata !{i32 %IntParI2}, i64 0, metadata !26), !dbg !67
  %add = add i32 %IntParI1, 5, !dbg !68
  tail call void @llvm.dbg.value(metadata !{i32 %add}, i64 0, metadata !27), !dbg !68
  %idxprom = sext i32 %add to i64, !dbg !69
  %arrayidx = getelementptr inbounds i32* %Array1Par, i64 %idxprom, !dbg !69
  store i32 %IntParI2, i32* %arrayidx, align 4, !dbg !69
  %add3 = add nsw i32 %IntParI1, 6, !dbg !73
  %idxprom4 = sext i32 %add3 to i64, !dbg !73
  %arrayidx5 = getelementptr inbounds i32* %Array1Par, i64 %idxprom4, !dbg !73
  store i32 %IntParI2, i32* %arrayidx5, align 4, !dbg !73
  %add6 = add nsw i32 %IntParI1, 35, !dbg !74
  %idxprom7 = sext i32 %add6 to i64, !dbg !74
  %arrayidx8 = getelementptr inbounds i32* %Array1Par, i64 %idxprom7, !dbg !74
  store i32 %add, i32* %arrayidx8, align 4, !dbg !74
  tail call void @llvm.dbg.value(metadata !{i32 %add}, i64 0, metadata !28), !dbg !75
  br label %for.body, !dbg !75

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %idxprom, %entry ], [ %indvars.iv.next, %for.body ]
  %IntIndex.046 = phi i32 [ %add, %entry ], [ %inc, %for.body ]
  %arrayidx13 = getelementptr inbounds [51 x i32]* %Array2Par, i64 %idxprom, i64 %indvars.iv, !dbg !77
  store i32 %add, i32* %arrayidx13, align 4, !dbg !77
  %inc = add nsw i32 %IntIndex.046, 1, !dbg !75
  tail call void @llvm.dbg.value(metadata !{i32 %inc}, i64 0, metadata !28), !dbg !75
  %cmp = icmp sgt i32 %inc, %add3, !dbg !75
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !75
  br i1 %cmp, label %for.end, label %for.body, !dbg !75

for.end:                                          ; preds = %for.body
  %sub = add nsw i32 %IntParI1, 4, !dbg !78
  %idxprom14 = sext i32 %sub to i64, !dbg !78
  %arrayidx17 = getelementptr inbounds [51 x i32]* %Array2Par, i64 %idxprom, i64 %idxprom14, !dbg !78
  %0 = load i32* %arrayidx17, align 4, !dbg !78
  %inc18 = add nsw i32 %0, 1, !dbg !78
  store i32 %inc18, i32* %arrayidx17, align 4, !dbg !78
  %1 = load i32* %arrayidx, align 4, !dbg !79
  %add22 = add nsw i32 %IntParI1, 25, !dbg !79
  %idxprom23 = sext i32 %add22 to i64, !dbg !79
  %arrayidx25 = getelementptr inbounds [51 x i32]* %Array2Par, i64 %idxprom23, i64 %idxprom, !dbg !79
  store i32 %1, i32* %arrayidx25, align 4, !dbg !79
  store i32 5, i32* @IntGlob, align 4, !dbg !80
  ret void, !dbg !81
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

attributes #0 = { nounwind optsize ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = metadata !{i32 786449, metadata !82, i32 12, metadata !"clang version 3.3 (trunk 175015)", i1 true, metadata !"", i32 0, metadata !1, metadata !10, metadata !11, metadata !29,  metadata !29, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/rdar_13183203/test2/dry.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 786436, metadata !82, null, metadata !"", i32 128, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 128, size 32, align 32, offset 0] [def] [from ]
!3 = metadata !{i32 786473, metadata !82} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !5, metadata !6, metadata !7, metadata !8, metadata !9}
!5 = metadata !{i32 786472, metadata !"Ident1", i64 0} ; [ DW_TAG_enumerator ] [Ident1 :: 0]
!6 = metadata !{i32 786472, metadata !"Ident2", i64 10000} ; [ DW_TAG_enumerator ] [Ident2 :: 10000]
!7 = metadata !{i32 786472, metadata !"Ident3", i64 10001} ; [ DW_TAG_enumerator ] [Ident3 :: 10001]
!8 = metadata !{i32 786472, metadata !"Ident4", i64 10002} ; [ DW_TAG_enumerator ] [Ident4 :: 10002]
!9 = metadata !{i32 786472, metadata !"Ident5", i64 10003} ; [ DW_TAG_enumerator ] [Ident5 :: 10003]
!10 = metadata !{i32 0}
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786478, metadata !82, metadata !3, metadata !"Proc8", metadata !"Proc8", metadata !"", i32 180, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, void (i32*, [51 x i32]*, i32, i32)* @Proc8, null, null, metadata !22, i32 185} ; [ DW_TAG_subprogram ] [line 180] [def] [scope 185] [Proc8]
!13 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{null, metadata !15, metadata !17, metadata !21, metadata !21}
!15 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!16 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!17 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !18} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!18 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 1632, i64 32, i32 0, i32 0, metadata !16, metadata !19, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 1632, align 32, offset 0] [from int]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786465, i64 0, i64 51}       ; [ DW_TAG_subrange_type ] [0, 50]
!21 = metadata !{i32 786454, metadata !82, null, metadata !"OneToFifty", i32 132, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ] [OneToFifty] [line 132, size 0, align 0, offset 0] [from int]
!22 = metadata !{metadata !23, metadata !24, metadata !25, metadata !26, metadata !27, metadata !28}
!23 = metadata !{i32 786689, metadata !12, metadata !"Array1Par", metadata !3, i32 16777397, metadata !15, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [Array1Par] [line 181]
!24 = metadata !{i32 786689, metadata !12, metadata !"Array2Par", metadata !3, i32 33554614, metadata !17, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [Array2Par] [line 182]
!25 = metadata !{i32 786689, metadata !12, metadata !"IntParI1", metadata !3, i32 50331831, metadata !21, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [IntParI1] [line 183]
!26 = metadata !{i32 786689, metadata !12, metadata !"IntParI2", metadata !3, i32 67109048, metadata !21, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [IntParI2] [line 184]
!27 = metadata !{i32 786688, metadata !12, metadata !"IntLoc", metadata !3, i32 186, metadata !21, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [IntLoc] [line 186]
!28 = metadata !{i32 786688, metadata !12, metadata !"IntIndex", metadata !3, i32 187, metadata !21, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [IntIndex] [line 187]
!29 = metadata !{metadata !30, metadata !35, metadata !36, metadata !38, metadata !39, metadata !40, metadata !42, metadata !46, metadata !63}
!30 = metadata !{i32 786484, i32 0, null, metadata !"Version", metadata !"Version", metadata !"", metadata !3, i32 111, metadata !31, i32 0, i32 1, [4 x i8]* @Version, null} ; [ DW_TAG_variable ] [Version] [line 111] [def]
!31 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 32, i64 8, i32 0, i32 0, metadata !32, metadata !33, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 8, offset 0] [from char]
!32 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!33 = metadata !{metadata !34}
!34 = metadata !{i32 786465, i64 0, i64 4}        ; [ DW_TAG_subrange_type ] [0, 3]
!35 = metadata !{i32 786484, i32 0, null, metadata !"IntGlob", metadata !"IntGlob", metadata !"", metadata !3, i32 171, metadata !16, i32 0, i32 1, i32* @IntGlob, null} ; [ DW_TAG_variable ] [IntGlob] [line 171] [def]
!36 = metadata !{i32 786484, i32 0, null, metadata !"BoolGlob", metadata !"BoolGlob", metadata !"", metadata !3, i32 172, metadata !37, i32 0, i32 1, i32* @BoolGlob, null} ; [ DW_TAG_variable ] [BoolGlob] [line 172] [def]
!37 = metadata !{i32 786454, metadata !82, null, metadata !"boolean", i32 149, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ] [boolean] [line 149, size 0, align 0, offset 0] [from int]
!38 = metadata !{i32 786484, i32 0, null, metadata !"Char1Glob", metadata !"Char1Glob", metadata !"", metadata !3, i32 173, metadata !32, i32 0, i32 1, i8* @Char1Glob, null} ; [ DW_TAG_variable ] [Char1Glob] [line 173] [def]
!39 = metadata !{i32 786484, i32 0, null, metadata !"Char2Glob", metadata !"Char2Glob", metadata !"", metadata !3, i32 174, metadata !32, i32 0, i32 1, i8* @Char2Glob, null} ; [ DW_TAG_variable ] [Char2Glob] [line 174] [def]
!40 = metadata !{i32 786484, i32 0, null, metadata !"Array1Glob", metadata !"Array1Glob", metadata !"", metadata !3, i32 175, metadata !41, i32 0, i32 1, [51 x i32]* @Array1Glob, null} ; [ DW_TAG_variable ] [Array1Glob] [line 175] [def]
!41 = metadata !{i32 786454, metadata !82, null, metadata !"Array1Dim", i32 135, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_typedef ] [Array1Dim] [line 135, size 0, align 0, offset 0] [from ]
!42 = metadata !{i32 786484, i32 0, null, metadata !"Array2Glob", metadata !"Array2Glob", metadata !"", metadata !3, i32 176, metadata !43, i32 0, i32 1, [51 x [51 x i32]]* @Array2Glob, null} ; [ DW_TAG_variable ] [Array2Glob] [line 176] [def]
!43 = metadata !{i32 786454, metadata !82, null, metadata !"Array2Dim", i32 136, i64 0, i64 0, i64 0, i32 0, metadata !44} ; [ DW_TAG_typedef ] [Array2Dim] [line 136, size 0, align 0, offset 0] [from ]
!44 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 83232, i64 32, i32 0, i32 0, metadata !16, metadata !45, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 83232, align 32, offset 0] [from int]
!45 = metadata !{metadata !20, metadata !20}
!46 = metadata !{i32 786484, i32 0, null, metadata !"PtrGlb", metadata !"PtrGlb", metadata !"", metadata !3, i32 177, metadata !47, i32 0, i32 1, %struct.Record** @PtrGlb, null} ; [ DW_TAG_variable ] [PtrGlb] [line 177] [def]
!47 = metadata !{i32 786454, metadata !82, null, metadata !"RecordPtr", i32 148, i64 0, i64 0, i64 0, i32 0, metadata !48} ; [ DW_TAG_typedef ] [RecordPtr] [line 148, size 0, align 0, offset 0] [from ]
!48 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !49} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from RecordType]
!49 = metadata !{i32 786454, metadata !82, null, metadata !"RecordType", i32 147, i64 0, i64 0, i64 0, i32 0, metadata !50} ; [ DW_TAG_typedef ] [RecordType] [line 147, size 0, align 0, offset 0] [from Record]
!50 = metadata !{i32 786451, metadata !82, null, metadata !"Record", i32 138, i64 448, i64 64, i32 0, i32 0, null, metadata !51, i32 0, null, i32 0, null} ; [ DW_TAG_structure_type ] [Record] [line 138, size 448, align 64, offset 0] [def] [from ]
!51 = metadata !{metadata !52, metadata !54, metadata !56, metadata !57, metadata !58}
!52 = metadata !{i32 786445, metadata !82, metadata !50, metadata !"PtrComp", i32 140, i64 64, i64 64, i64 0, i32 0, metadata !53} ; [ DW_TAG_member ] [PtrComp] [line 140, size 64, align 64, offset 0] [from ]
!53 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !50} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Record]
!54 = metadata !{i32 786445, metadata !82, metadata !50, metadata !"Discr", i32 141, i64 32, i64 32, i64 64, i32 0, metadata !55} ; [ DW_TAG_member ] [Discr] [line 141, size 32, align 32, offset 64] [from Enumeration]
!55 = metadata !{i32 786454, metadata !82, null, metadata !"Enumeration", i32 128, i64 0, i64 0, i64 0, i32 0, metadata !2} ; [ DW_TAG_typedef ] [Enumeration] [line 128, size 0, align 0, offset 0] [from ]
!56 = metadata !{i32 786445, metadata !82, metadata !50, metadata !"EnumComp", i32 142, i64 32, i64 32, i64 96, i32 0, metadata !55} ; [ DW_TAG_member ] [EnumComp] [line 142, size 32, align 32, offset 96] [from Enumeration]
!57 = metadata !{i32 786445, metadata !82, metadata !50, metadata !"IntComp", i32 143, i64 32, i64 32, i64 128, i32 0, metadata !21} ; [ DW_TAG_member ] [IntComp] [line 143, size 32, align 32, offset 128] [from OneToFifty]
!58 = metadata !{i32 786445, metadata !82, metadata !50, metadata !"StringComp", i32 144, i64 248, i64 8, i64 160, i32 0, metadata !59} ; [ DW_TAG_member ] [StringComp] [line 144, size 248, align 8, offset 160] [from String30]
!59 = metadata !{i32 786454, metadata !82, null, metadata !"String30", i32 134, i64 0, i64 0, i64 0, i32 0, metadata !60} ; [ DW_TAG_typedef ] [String30] [line 134, size 0, align 0, offset 0] [from ]
!60 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 248, i64 8, i32 0, i32 0, metadata !32, metadata !61, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 248, align 8, offset 0] [from char]
!61 = metadata !{metadata !62}
!62 = metadata !{i32 786465, i64 0, i64 31}       ; [ DW_TAG_subrange_type ] [0, 30]
!63 = metadata !{i32 786484, i32 0, null, metadata !"PtrGlbNext", metadata !"PtrGlbNext", metadata !"", metadata !3, i32 178, metadata !47, i32 0, i32 1, %struct.Record** @PtrGlbNext, null} ; [ DW_TAG_variable ] [PtrGlbNext] [line 178] [def]
!64 = metadata !{i32 181, i32 0, metadata !12, null}
!65 = metadata !{i32 182, i32 0, metadata !12, null}
!66 = metadata !{i32 183, i32 0, metadata !12, null}
!67 = metadata !{i32 184, i32 0, metadata !12, null}
!68 = metadata !{i32 189, i32 0, metadata !12, null}
!69 = metadata !{i32 190, i32 0, metadata !12, null}
!73 = metadata !{i32 191, i32 0, metadata !12, null}
!74 = metadata !{i32 192, i32 0, metadata !12, null}
!75 = metadata !{i32 193, i32 0, metadata !76, null}
!76 = metadata !{i32 786443, metadata !82, metadata !12, i32 193, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/Users/manmanren/test-Nov/rdar_13183203/test2/dry.c]
!77 = metadata !{i32 194, i32 0, metadata !76, null}
!78 = metadata !{i32 195, i32 0, metadata !12, null}
!79 = metadata !{i32 196, i32 0, metadata !12, null}
!80 = metadata !{i32 197, i32 0, metadata !12, null}
!81 = metadata !{i32 198, i32 0, metadata !12, null}
!82 = metadata !{metadata !"dry.c", metadata !"/Users/manmanren/test-Nov/rdar_13183203/test2"}
!83 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
