; RUN: llc -mtriple=x86_64-apple-darwin < %s -o -

; PR16954
;
; Make sure that when we splice off the end of a machine basic block, we include
; DBG_VALUE MI in the terminator sequence.

@a = external global { i64, [56 x i8] }, align 32

; Function Attrs: nounwind sspreq
define i32 @_Z18read_response_sizev() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !22, i64 0, metadata !23), !dbg !39
  %0 = load i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0), align 8, !dbg !40
  tail call void @llvm.dbg.value(metadata !63, i64 0, metadata !64), !dbg !71
  %1 = trunc i64 %0 to i32
  ret i32 %1
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata)

attributes #0 = { sspreq }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !72}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 true, metadata !"", i32 0, metadata !2, metadata !5, metadata !8, metadata !20, metadata !5, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/matt/ryan_bug/<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !"/Users/matt/ryan_bug"}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786436, metadata !1, metadata !4, metadata !"", i32 20, i64 32, i64 32, i32 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 19, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [C] [line 19, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{i32 0}
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786472, metadata !"max_frame_size", i64 0} ; [ DW_TAG_enumerator ] [max_frame_size :: 0]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"read_response_size", metadata !"read_response_size", metadata !"_Z18read_response_sizev", i32 27, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* @_Z18read_response_sizev, null, null, metadata !14, i32 27} ; [ DW_TAG_subprogram ] [line 27] [def] [read_response_size]
!10 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/Users/matt/ryan_bug/<unknown>]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !15, metadata !19}
!15 = metadata !{i32 786688, metadata !9, metadata !"b", metadata !10, i32 28, metadata !16, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 28]
!16 = metadata !{i32 786451, metadata !1, null, metadata !"B", i32 16, i64 32, i64 32, i32 0, i32 0, null, metadata !17, i32 0, null, null} ; [ DW_TAG_structure_type ] [B] [line 16, size 32, align 32, offset 0] [def] [from ]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786445, metadata !1, metadata !16, metadata !"end_of_file", i32 17, i64 32, i64 32, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ] [end_of_file] [line 17, size 32, align 32, offset 0] [from int]
!19 = metadata !{i32 786688, metadata !9, metadata !"c", metadata !10, i32 29, metadata !13, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [c] [line 29]
!20 = metadata !{}
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0)}
!23 = metadata !{i32 786689, metadata !24, metadata !"p2", metadata !10, i32 33554444, metadata !32, i32 0, metadata !38} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!24 = metadata !{i32 786478, metadata !1, metadata !25, metadata !"min<unsigned long long>", metadata !"min<unsigned long long>", metadata !"_ZN3__13minIyEERKT_S3_RS1_", i32 12, metadata !27, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, metadata !33, null, metadata !35, i32 12} ; [ DW_TAG_subprogram ] [line 12] [def] [min<unsigned long long>]
!25 = metadata !{i32 786489, metadata !26, null, metadata !"__1", i32 1} ; [ DW_TAG_namespace ] [__1] [line 1]
!26 = metadata !{metadata !"main.cpp", metadata !"/Users/matt/ryan_bug"}
!27 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !28, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{metadata !29, metadata !29, metadata !32}
!29 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !30} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !31} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!31 = metadata !{i32 786468, null, null, metadata !"long long unsigned int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [long long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!32 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !31} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!33 = metadata !{metadata !34}
!34 = metadata !{i32 786479, null, metadata !"_Tp", metadata !31, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!35 = metadata !{metadata !36, metadata !37}
!36 = metadata !{i32 786689, metadata !24, metadata !"p1", metadata !10, i32 16777228, metadata !29, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p1] [line 12]
!37 = metadata !{i32 786689, metadata !24, metadata !"p2", metadata !10, i32 33554444, metadata !32, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!38 = metadata !{i32 33, i32 0, metadata !9, null}
!39 = metadata !{i32 12, i32 0, metadata !24, metadata !38}
!40 = metadata !{i32 9, i32 0, metadata !41, metadata !59}
!41 = metadata !{i32 786478, metadata !1, metadata !25, metadata !"min<unsigned long long, __1::A>", metadata !"min<unsigned long long, __1::A>", metadata !"_ZN3__13minIyNS_1AEEERKT_S4_RS2_T0_", i32 7, metadata !42, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, metadata !53, null, metadata !55, i32 8} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [min<unsigned long long, __1::A>]
!42 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !43, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = metadata !{metadata !29, metadata !29, metadata !32, metadata !44}
!44 = metadata !{i32 786451, metadata !1, metadata !25, metadata !"A", i32 0, i64 8, i64 8, i32 0, i32 0, null, metadata !45, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 0, size 8, align 8, offset 0] [def] [from ]
!45 = metadata !{metadata !46}
!46 = metadata !{i32 786478, metadata !1, metadata !44, metadata !"operator()", metadata !"operator()", metadata !"_ZN3__11AclERKiS2_", i32 1, metadata !47, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !52, i32 1} ; [ DW_TAG_subprogram ] [line 1] [operator()]
!47 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !48, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!48 = metadata !{metadata !13, metadata !49, metadata !50, metadata !50}
!49 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!50 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !51} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!51 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!52 = metadata !{i32 786468}
!53 = metadata !{metadata !34, metadata !54}
!54 = metadata !{i32 786479, null, metadata !"_Compare", metadata !44, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!55 = metadata !{metadata !56, metadata !57, metadata !58}
!56 = metadata !{i32 786689, metadata !41, metadata !"p1", metadata !10, i32 16777223, metadata !29, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p1] [line 7]
!57 = metadata !{i32 786689, metadata !41, metadata !"p2", metadata !10, i32 33554439, metadata !32, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p2] [line 7]
!58 = metadata !{i32 786689, metadata !41, metadata !"p3", metadata !10, i32 50331656, metadata !44, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p3] [line 8]
!59 = metadata !{i32 13, i32 0, metadata !24, metadata !38}
!63 = metadata !{i32 undef}
!64 = metadata !{i32 786689, metadata !65, metadata !"p1", metadata !10, i32 33554433, metadata !50, i32 0, metadata !40} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!65 = metadata !{i32 786478, metadata !1, metadata !25, metadata !"operator()", metadata !"operator()", metadata !"_ZN3__11AclERKiS2_", i32 1, metadata !47, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !46, metadata !66, i32 2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [operator()]
!66 = metadata !{metadata !67, metadata !69, metadata !70}
!67 = metadata !{i32 786689, metadata !65, metadata !"this", null, i32 16777216, metadata !68, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!69 = metadata !{i32 786689, metadata !65, metadata !"p1", metadata !10, i32 33554433, metadata !50, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!70 = metadata !{i32 786689, metadata !65, metadata !"", metadata !10, i32 50331650, metadata !50, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 2]
!71 = metadata !{i32 1, i32 0, metadata !65, metadata !40}
!72 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
