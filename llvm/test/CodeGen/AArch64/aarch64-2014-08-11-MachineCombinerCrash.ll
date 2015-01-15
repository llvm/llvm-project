; RUN: llc < %s -O2 -mtriple=aarch64-none-linux-gnu 

; Bug 20598


define void @test() #0 {
entry:
  br label %for.body, !dbg !39

for.body:                                         ; preds = %for.body, %entry
  %arrayidx5 = getelementptr inbounds i32* null, i64 1, !dbg !43
  %0 = load i32* null, align 4, !dbg !45, !tbaa !46
  %s1 = sub nsw i32 0, %0, !dbg !50
  %n1 = sext i32 %s1 to i64, !dbg !50
  %arrayidx21 = getelementptr inbounds i32* null, i64 3, !dbg !51
  %add53 = add nsw i64 %n1, 0, !dbg !52
  %add55 = add nsw i64 %n1, 0, !dbg !53
  %mul63 = mul nsw i64 %add53, -20995, !dbg !54
  tail call void @llvm.dbg.value(metadata i64 %mul63, i64 0, metadata !30, metadata !{!"0x102"}), !dbg !55
  %mul65 = mul nsw i64 %add55, -3196, !dbg !56
  %add67 = add nsw i64 0, %mul65, !dbg !57
  %add80 = add i64 0, 1024, !dbg !58
  %add81 = add i64 %add80, %mul63, !dbg !58
  %add82 = add i64 %add81, 0, !dbg !58
  %shr83351 = lshr i64 %add82, 11, !dbg !58
  %conv84 = trunc i64 %shr83351 to i32, !dbg !58
  store i32 %conv84, i32* %arrayidx21, align 4, !dbg !58, !tbaa !46
  %add86 = add i64 0, 1024, !dbg !59
  %add87 = add i64 %add86, 0, !dbg !59
  %add88 = add i64 %add87, %add67, !dbg !59
  %shr89352 = lshr i64 %add88, 11, !dbg !59
  %n2 = trunc i64 %shr89352 to i32, !dbg !59
  store i32 %n2, i32* %arrayidx5, align 4, !dbg !59, !tbaa !46
  br label %for.body, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}
!llvm.ident = !{!38}

!0 = !{!"0x11\0012\00clang version 3.6.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [] [] []
!1 = !{!"test.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00\00\00\00140\000\001\000\006\00256\001\00141", !1, !5, !6, null, void ()* @test, null, null, !12} ; [ DW_TAG_subprogram ] [] [] [def] [scope 141] []
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ] [] []
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [] [] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [] [] []
!9 = !{!"0x16\00\0030\000\000\000\000", !10, null, !11} ; [ DW_TAG_typedef ] [] [] [] [from int]
!10 = !{!"", !""}
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [] [int] []
!12 = !{!13, !14, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35}
!13 = !{!"0x101\00\0016777356\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [] [data] []
!14 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!15 = !{!"0x16\00\00183\000\000\000\000", !16, null, !17} ; [ DW_TAG_typedef ] [] [INT32] [] [from long int]
!16 = !{!"", !""}
!17 = !{!"0x24\00\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [] [long int] []
!18 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!19 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!20 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!21 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!22 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!23 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [] [] []
!24 = !{!"0x100\00\00142\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!25 = !{!"0x100\00\00143\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!26 = !{!"0x100\00\00143\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!27 = !{!"0x100\00\00143\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!28 = !{!"0x100\00\00143\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!29 = !{!"0x100\00\00144\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!30 = !{!"0x100\00\00144\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!31 = !{!"0x100\00\00144\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!32 = !{!"0x100\00\00144\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [ ] [] []
!33 = !{!"0x100\00\00144\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!34 = !{!"0x100\00\00145\000", !4, !5, !8} ; [ DW_TAG_auto_variable ] [  ] [] []
!35 = !{!"0x100\00\00146\000", !4, !5, !11} ; [ DW_TAG_auto_variable ] [  ] [] []
!36 = !{i32 2, !"Dwarf Version", i32 4}
!37 = !{i32 2, !"Debug Info Version", i32 2}
!38 = !{!"clang version 3.6.0 "}
!39 = !MDLocation(line: 154, column: 8, scope: !40)
!40 = !{!"0xb\00154\008\002", !1, !41} ; [ DW_TAG_lexical_block ] [  ] []
!41 = !{!"0xb\00154\008\001", !1, !42} ; [ DW_TAG_lexical_block ] [  ] []
!42 = !{!"0xb\00154\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [  ] []
!43 = !MDLocation(line: 157, column: 5, scope: !44)
!44 = !{!"0xb\00154\0042\000", !1, !42} ; [ DW_TAG_lexical_block ] [  ] []
!45 = !MDLocation(line: 159, column: 5, scope: !44)
!46 = !{!47, !47, i64 0}
!47 = !{!"int", !48, i64 0}
!48 = !{!"omnipotent char", !49, i64 0}
!49 = !{!"Simple C/C++ TBAA"}
!50 = !MDLocation(line: 160, column: 5, scope: !44)
!51 = !MDLocation(line: 161, column: 5, scope: !44)
!52 = !MDLocation(line: 188, column: 5, scope: !44)
!53 = !MDLocation(line: 190, column: 5, scope: !44)
!54 = !MDLocation(line: 198, column: 5, scope: !44)
!55 = !MDLocation(line: 144, column: 13, scope: !4)
!56 = !MDLocation(line: 200, column: 5, scope: !44)
!57 = !MDLocation(line: 203, column: 5, scope: !44)
!58 = !MDLocation(line: 207, column: 5, scope: !44)
!59 = !MDLocation(line: 208, column: 5, scope: !44)
