; RUN: llc -mtriple=x86_64-apple-darwin < %s -o -

; PR16954
;
; Make sure that when we splice off the end of a machine basic block, we include
; DBG_VALUE MI in the terminator sequence.

@a = external global { i64, [56 x i8] }, align 32

; Function Attrs: nounwind sspreq
define i32 @_Z18read_response_sizev() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !22, i64 0, metadata !23, metadata !{!"0x102"}), !dbg !39
  %0 = load i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0), align 8, !dbg !40
  tail call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !64, metadata !{!"0x102"}), !dbg !71
  %1 = trunc i64 %0 to i32
  ret i32 %1
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { sspreq }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !72}

!0 = !{!"0x11\004\00clang version 3.4 \001\00\000\00\001", !1, !2, !5, !8, !20, !5} ; [ DW_TAG_compile_unit ] [/Users/matt/ryan_bug/<unknown>] [DW_LANG_C_plus_plus]
!1 = !{!"<unknown>", !"/Users/matt/ryan_bug"}
!2 = !{!3}
!3 = !{!"0x4\00\0020\0032\0032\000\000\000", !1, !4, null, !6, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!4 = !{!"0x13\00C\0019\008\008\000\000\000", !1, null, null, !5, null, null, null} ; [ DW_TAG_structure_type ] [C] [line 19, size 8, align 8, offset 0] [def] [from ]
!5 = !{}
!6 = !{!7}
!7 = !{!"0x28\00max_frame_size\000"} ; [ DW_TAG_enumerator ] [max_frame_size :: 0]
!8 = !{!9, !24, !41, !65}
!9 = !{!"0x2e\00read_response_size\00read_response_size\00_Z18read_response_sizev\0027\000\001\000\006\00256\001\0027", !1, !10, !11, null, i32 ()* @_Z18read_response_sizev, null, null, !14} ; [ DW_TAG_subprogram ] [line 27] [def] [read_response_size]
!10 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/Users/matt/ryan_bug/<unknown>]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!13}
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !{!15, !19}
!15 = !{!"0x100\00b\0028\000", !9, !10, !16} ; [ DW_TAG_auto_variable ] [b] [line 28]
!16 = !{!"0x13\00B\0016\0032\0032\000\000\000", !1, null, null, !17, null, null} ; [ DW_TAG_structure_type ] [B] [line 16, size 32, align 32, offset 0] [def] [from ]
!17 = !{!18}
!18 = !{!"0xd\00end_of_file\0017\0032\0032\000\000", !1, !16, !13} ; [ DW_TAG_member ] [end_of_file] [line 17, size 32, align 32, offset 0] [from int]
!19 = !{!"0x100\00c\0029\000", !9, !10, !13} ; [ DW_TAG_auto_variable ] [c] [line 29]
!20 = !{}
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i64* getelementptr inbounds ({ i64, [56 x i8] }* @a, i32 0, i32 0)}
!23 = !{!"0x101\00p2\0033554444\000", !24, !10, !32, !38} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!24 = !{!"0x2e\00min<unsigned long long>\00min<unsigned long long>\00_ZN3__13minIyEERKT_S3_RS1_\0012\000\001\000\006\00256\001\0012", !1, !25, !27, null, null, !33, null, !35} ; [ DW_TAG_subprogram ] [line 12] [def] [min<unsigned long long>]
!25 = !{!"0x39\00__1\001", !26, null} ; [ DW_TAG_namespace ] [__1] [line 1]
!26 = !{!"main.cpp", !"/Users/matt/ryan_bug"}
!27 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !28, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = !{!29, !29, !32}
!29 = !{!"0x10\00\000\000\000\000\000", null, null, !30} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!"0x26\00\000\000\000\000\000", null, null, !31} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!31 = !{!"0x24\00long long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!32 = !{!"0x10\00\000\000\000\000\000", null, null, !31} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!33 = !{!34}
!34 = !{!"0x2f\00_Tp\000\000", null, !31, null} ; [ DW_TAG_template_type_parameter ]
!35 = !{!36, !37}
!36 = !{!"0x101\00p1\0016777228\000", !24, !10, !29} ; [ DW_TAG_arg_variable ] [p1] [line 12]
!37 = !{!"0x101\00p2\0033554444\000", !24, !10, !32} ; [ DW_TAG_arg_variable ] [p2] [line 12]
!38 = !MDLocation(line: 33, scope: !9)
!39 = !MDLocation(line: 12, scope: !24, inlinedAt: !38)
!40 = !MDLocation(line: 9, scope: !41, inlinedAt: !59)
!41 = !{!"0x2e\00min<unsigned long long, __1::A>\00min<unsigned long long, __1::A>\00_ZN3__13minIyNS_1AEEERKT_S4_RS2_T0_\007\000\001\000\006\00256\001\008", !1, !25, !42, null, null, !53, null, !55} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [min<unsigned long long, __1::A>]
!42 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !43, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = !{!29, !29, !32, !44}
!44 = !{!"0x13\00A\000\008\008\000\000\000", !1, !25, null, !45, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 0, size 8, align 8, offset 0] [def] [from ]
!45 = !{!46}
!46 = !{!"0x2e\00operator()\00operator()\00_ZN3__11AclERKiS2_\001\000\000\000\006\00256\001\001", !1, !44, !47, null, null, null, i32 0, !52} ; [ DW_TAG_subprogram ] [line 1] [operator()]
!47 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !48, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!48 = !{!13, !49, !50, !50}
!49 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!50 = !{!"0x10\00\000\000\000\000\000", null, null, !51} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!51 = !{!"0x26\00\000\000\000\000\000", null, null, !13} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!52 = !{i32 786468}
!53 = !{!34, !54}
!54 = !{!"0x2f\00_Compare\000\000", null, !44, null} ; [ DW_TAG_template_type_parameter ]
!55 = !{!56, !57, !58}
!56 = !{!"0x101\00p1\0016777223\000", !41, !10, !29} ; [ DW_TAG_arg_variable ] [p1] [line 7]
!57 = !{!"0x101\00p2\0033554439\000", !41, !10, !32} ; [ DW_TAG_arg_variable ] [p2] [line 7]
!58 = !{!"0x101\00p3\0050331656\000", !41, !10, !44} ; [ DW_TAG_arg_variable ] [p3] [line 8]
!59 = !MDLocation(line: 13, scope: !24, inlinedAt: !38)
!63 = !{i32 undef}
!64 = !{!"0x101\00p1\0033554433\000", !65, !10, !50, !40} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!65 = !{!"0x2e\00operator()\00operator()\00_ZN3__11AclERKiS2_\001\000\001\000\006\00256\001\002", !1, !25, !47, null, null, null, !46, !66} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [operator()]
!66 = !{!67, !69, !70}
!67 = !{!"0x101\00this\0016777216\001088", !65, null, !68} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = !{!"0xf\00\000\0064\0064\000\000", null, null, !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!69 = !{!"0x101\00p1\0033554433\000", !65, !10, !50} ; [ DW_TAG_arg_variable ] [p1] [line 1]
!70 = !{!"0x101\00\0050331650\000", !65, !10, !50} ; [ DW_TAG_arg_variable ] [line 2]
!71 = !MDLocation(line: 1, scope: !65, inlinedAt: !40)
!72 = !{i32 1, !"Debug Info Version", i32 2}
