; RUN: llc < %s | FileCheck %s

; Check debug info output for merged global.
; DW_AT_location
; DW_OP_addr
; DW_OP_plus
; .long __MergedGlobals
; DW_OP_constu
; offset

;CHECK: .long Lset8
;CHECK-NEXT:        @ DW_AT_type
;CHECK-NEXT:        @ DW_AT_decl_file
;CHECK-NEXT:        @ DW_AT_decl_line
;CHECK-NEXT:        @ DW_AT_location
;CHECK-NEXT:        .byte   3
;CHECK-NEXT:        .long   __MergedGlobals
;CHECK-NEXT:        .byte   16
; 4 is byte offset of x2 in __MergedGobals
;CHECK-NEXT:        .byte   4
;CHECK-NEXT:        .byte   34


target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.7.0"

@x1 = internal unnamed_addr global i32 1, align 4
@x2 = internal unnamed_addr global i32 2, align 4
@x3 = internal unnamed_addr global i32 3, align 4
@x4 = internal unnamed_addr global i32 4, align 4
@x5 = global i32 0, align 4

define i32 @get1(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !10), !dbg !30
  %1 = load i32* @x1, align 4, !dbg !31
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !11), !dbg !31
  store i32 %a, i32* @x1, align 4, !dbg !31
  ret i32 %1, !dbg !31
}

define i32 @get2(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !13), !dbg !32
  %1 = load i32* @x2, align 4, !dbg !33
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !14), !dbg !33
  store i32 %a, i32* @x2, align 4, !dbg !33
  ret i32 %1, !dbg !33
}

define i32 @get3(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !16), !dbg !34
  %1 = load i32* @x3, align 4, !dbg !35
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !17), !dbg !35
  store i32 %a, i32* @x3, align 4, !dbg !35
  ret i32 %1, !dbg !35
}

define i32 @get4(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !19), !dbg !36
  %1 = load i32* @x4, align 4, !dbg !37
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !20), !dbg !37
  store i32 %a, i32* @x4, align 4, !dbg !37
  ret i32 %1, !dbg !37
}

define i32 @get5(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !27), !dbg !38
  %1 = load i32* @x5, align 4, !dbg !39
  tail call void @llvm.dbg.value(metadata !{i32 %1}, i64 0, metadata !28), !dbg !39
  store i32 %a, i32* @x5, align 4, !dbg !39
  ret i32 %1, !dbg !39
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!49}

!0 = metadata !{i32 786449, metadata !47, i32 12, metadata !"clang", i1 true, metadata !"", i32 0, metadata !48, metadata !48, metadata !40, metadata !41,  metadata !41, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 786478, metadata !47, metadata !2, metadata !"get1", metadata !"get1", metadata !"", i32 5, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @get1, null, null, metadata !42, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [get1]
!2 = metadata !{i32 786473, metadata !47} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786453, metadata !47, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786478, metadata !47, metadata !2, metadata !"get2", metadata !"get2", metadata !"", i32 8, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @get2, null, null, metadata !43, i32 8} ; [ DW_TAG_subprogram ] [line 8] [def] [get2]
!7 = metadata !{i32 786478, metadata !47, metadata !2, metadata !"get3", metadata !"get3", metadata !"", i32 11, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @get3, null, null, metadata !44, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [get3]
!8 = metadata !{i32 786478, metadata !47, metadata !2, metadata !"get4", metadata !"get4", metadata !"", i32 14, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @get4, null, null, metadata !45, i32 14} ; [ DW_TAG_subprogram ] [line 14] [def] [get4]
!9 = metadata !{i32 786478, metadata !47, metadata !2, metadata !"get5", metadata !"get5", metadata !"", i32 17, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @get5, null, null, metadata !46, i32 17} ; [ DW_TAG_subprogram ] [line 17] [def] [get5]
!10 = metadata !{i32 786689, metadata !1, metadata !"a", metadata !2, i32 16777221, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 786688, metadata !12, metadata !"b", metadata !2, i32 5, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!12 = metadata !{i32 786443, metadata !47, metadata !1, i32 5, i32 19, i32 0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 786689, metadata !6, metadata !"a", metadata !2, i32 16777224, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!14 = metadata !{i32 786688, metadata !15, metadata !"b", metadata !2, i32 8, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 786443, metadata !47, metadata !6, i32 8, i32 17, i32 1} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 786689, metadata !7, metadata !"a", metadata !2, i32 16777227, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 786688, metadata !18, metadata !"b", metadata !2, i32 11, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 786443, metadata !47, metadata !7, i32 11, i32 19, i32 2} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 786689, metadata !8, metadata !"a", metadata !2, i32 16777230, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 786688, metadata !21, metadata !"b", metadata !2, i32 14, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!21 = metadata !{i32 786443, metadata !47, metadata !8, i32 14, i32 19, i32 3} ; [ DW_TAG_lexical_block ]
!22 = metadata !{i32 786484, i32 0, metadata !0, metadata !"x5", metadata !"x5", metadata !"", metadata !2, i32 16, metadata !5, i32 0, i32 1, i32* @x5, null} ; [ DW_TAG_variable ]
!23 = metadata !{i32 786484, i32 0, metadata !0, metadata !"x4", metadata !"x4", metadata !"", metadata !2, i32 13, metadata !5, i32 1, i32 1, i32* @x4, null} ; [ DW_TAG_variable ]
!24 = metadata !{i32 786484, i32 0, metadata !0, metadata !"x3", metadata !"x3", metadata !"", metadata !2, i32 10, metadata !5, i32 1, i32 1, i32* @x3, null} ; [ DW_TAG_variable ]
!25 = metadata !{i32 786484, i32 0, metadata !0, metadata !"x2", metadata !"x2", metadata !"", metadata !2, i32 7, metadata !5, i32 1, i32 1, i32* @x2, null} ; [ DW_TAG_variable ]
!26 = metadata !{i32 786484, i32 0, metadata !0, metadata !"x1", metadata !"x1", metadata !"", metadata !2, i32 4, metadata !5, i32 1, i32 1, i32* @x1, null} ; [ DW_TAG_variable ]
!27 = metadata !{i32 786689, metadata !9, metadata !"a", metadata !2, i32 16777233, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!28 = metadata !{i32 786688, metadata !29, metadata !"b", metadata !2, i32 17, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 786443, metadata !47, metadata !9, i32 17, i32 19, i32 4} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 5, i32 16, metadata !1, null}
!31 = metadata !{i32 5, i32 32, metadata !12, null}
!32 = metadata !{i32 8, i32 14, metadata !6, null}
!33 = metadata !{i32 8, i32 29, metadata !15, null}
!34 = metadata !{i32 11, i32 16, metadata !7, null}
!35 = metadata !{i32 11, i32 32, metadata !18, null}
!36 = metadata !{i32 14, i32 16, metadata !8, null}
!37 = metadata !{i32 14, i32 32, metadata !21, null}
!38 = metadata !{i32 17, i32 16, metadata !9, null}
!39 = metadata !{i32 17, i32 32, metadata !29, null}
!40 = metadata !{metadata !1, metadata !6, metadata !7, metadata !8, metadata !9}
!41 = metadata !{metadata !22, metadata !23, metadata !24, metadata !25, metadata !26}
!42 = metadata !{metadata !10, metadata !11}
!43 = metadata !{metadata !13, metadata !14}
!44 = metadata !{metadata !16, metadata !17}
!45 = metadata !{metadata !19, metadata !20}
!46 = metadata !{metadata !27, metadata !28}
!47 = metadata !{metadata !"ss3.c", metadata !"/private/tmp"}
!48 = metadata !{i32 0}
!49 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
