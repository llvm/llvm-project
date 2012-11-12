; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@x1 = internal global i8 1, align 1
@x2 = internal global i8 1, align 1
@x3 = internal global i8 1, align 1
@x4 = internal global i8 1, align 1
@x5 = global i8 1, align 1

; Check debug info output for merged global.
; DW_AT_location
; DW_OP_addr
; DW_OP_plus
; .long __MergedGlobals
; DW_OP_constu
; offset

;CHECK: .long Lset6
;CHECK-NEXT:        @ DW_AT_type
;CHECK-NEXT:        @ DW_AT_decl_file
;CHECK-NEXT:        @ DW_AT_decl_line
;CHECK-NEXT:        @ DW_AT_location
;CHECK-NEXT:        .byte   3
;CHECK-NEXT:        .long   __MergedGlobals
;CHECK-NEXT:        .byte   16
;CHECK-NEXT:        .byte   1
;CHECK-NEXT:        .byte   34

define zeroext i8 @get1(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8 %a}, i64 0, metadata !10), !dbg !30
  %0 = load i8* @x1, align 4, !dbg !30
  tail call void @llvm.dbg.value(metadata !{i8 %0}, i64 0, metadata !11), !dbg !30
  store i8 %a, i8* @x1, align 4, !dbg !30
  ret i8 %0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define zeroext i8 @get2(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8 %a}, i64 0, metadata !18), !dbg !32
  %0 = load i8* @x2, align 4, !dbg !32
  tail call void @llvm.dbg.value(metadata !{i8 %0}, i64 0, metadata !19), !dbg !32
  store i8 %a, i8* @x2, align 4, !dbg !32
  ret i8 %0, !dbg !33
}

define zeroext i8 @get3(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8 %a}, i64 0, metadata !21), !dbg !34
  %0 = load i8* @x3, align 4, !dbg !34
  tail call void @llvm.dbg.value(metadata !{i8 %0}, i64 0, metadata !22), !dbg !34
  store i8 %a, i8* @x3, align 4, !dbg !34
  ret i8 %0, !dbg !35
}

define zeroext i8 @get4(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8 %a}, i64 0, metadata !24), !dbg !36
  %0 = load i8* @x4, align 4, !dbg !36
  tail call void @llvm.dbg.value(metadata !{i8 %0}, i64 0, metadata !25), !dbg !36
  store i8 %a, i8* @x4, align 4, !dbg !36
  ret i8 %0, !dbg !37
}

define zeroext i8 @get5(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8 %a}, i64 0, metadata !27), !dbg !38
  %0 = load i8* @x5, align 4, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i8 %0}, i64 0, metadata !28), !dbg !38
  store i8 %a, i8* @x5, align 4, !dbg !38
  ret i8 %0, !dbg !39
}

!llvm.dbg.sp = !{!0, !6, !7, !8, !9}
!llvm.dbg.lv.get1 = !{!10, !11}
!llvm.dbg.gv = !{!13, !14, !15, !16, !17}
!llvm.dbg.lv.get2 = !{!18, !19}
!llvm.dbg.lv.get3 = !{!21, !22}
!llvm.dbg.lv.get4 = !{!24, !25}
!llvm.dbg.lv.get5 = !{!27, !28}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"get1", metadata !"get1", metadata !"get1", metadata !1, i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i8 (i8)* @get1} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"foo.c", metadata !"/tmp/", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"foo.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2369.8)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5, metadata !5}
!5 = metadata !{i32 589860, metadata !1, metadata !"_Bool", metadata !1, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !1, metadata !"get2", metadata !"get2", metadata !"get2", metadata !1, i32 7, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i8 (i8)* @get2} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589870, i32 0, metadata !1, metadata !"get3", metadata !"get3", metadata !"get3", metadata !1, i32 10, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i8 (i8)* @get3} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 589870, i32 0, metadata !1, metadata !"get4", metadata !"get4", metadata !"get4", metadata !1, i32 13, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i8 (i8)* @get4} ; [ DW_TAG_subprogram ]
!9 = metadata !{i32 589870, i32 0, metadata !1, metadata !"get5", metadata !"get5", metadata !"get5", metadata !1, i32 16, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i8 (i8)* @get5} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 590081, metadata !0, metadata !"a", metadata !1, i32 4, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 590080, metadata !12, metadata !"b", metadata !1, i32 4, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!12 = metadata !{i32 589835, metadata !0, i32 4, i32 0, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 589876, i32 0, metadata !1, metadata !"x1", metadata !"x1", metadata !"", metadata !1, i32 3, metadata !5, i1 true, i1 true, i8* @x1} ; [ DW_TAG_variable ]
!14 = metadata !{i32 589876, i32 0, metadata !1, metadata !"x2", metadata !"x2", metadata !"", metadata !1, i32 6, metadata !5, i1 true, i1 true, i8* @x2} ; [ DW_TAG_variable ]
!15 = metadata !{i32 589876, i32 0, metadata !1, metadata !"x3", metadata !"x3", metadata !"", metadata !1, i32 9, metadata !5, i1 true, i1 true, i8* @x3} ; [ DW_TAG_variable ]
!16 = metadata !{i32 589876, i32 0, metadata !1, metadata !"x4", metadata !"x4", metadata !"", metadata !1, i32 12, metadata !5, i1 true, i1 true, i8* @x4} ; [ DW_TAG_variable ]
!17 = metadata !{i32 589876, i32 0, metadata !1, metadata !"x5", metadata !"x5", metadata !"", metadata !1, i32 15, metadata !5, i1 false, i1 true, i8* @x5} ; [ DW_TAG_variable ]
!18 = metadata !{i32 590081, metadata !6, metadata !"a", metadata !1, i32 7, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 590080, metadata !20, metadata !"b", metadata !1, i32 7, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!20 = metadata !{i32 589835, metadata !6, i32 7, i32 0, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!21 = metadata !{i32 590081, metadata !7, metadata !"a", metadata !1, i32 10, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 590080, metadata !23, metadata !"b", metadata !1, i32 10, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!23 = metadata !{i32 589835, metadata !7, i32 10, i32 0, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!24 = metadata !{i32 590081, metadata !8, metadata !"a", metadata !1, i32 13, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 590080, metadata !26, metadata !"b", metadata !1, i32 13, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!26 = metadata !{i32 589835, metadata !8, i32 13, i32 0, metadata !1, i32 3} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 590081, metadata !9, metadata !"a", metadata !1, i32 16, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!28 = metadata !{i32 590080, metadata !29, metadata !"b", metadata !1, i32 16, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 589835, metadata !9, i32 16, i32 0, metadata !1, i32 4} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 4, i32 0, metadata !0, null}
!31 = metadata !{i32 4, i32 0, metadata !12, null}
!32 = metadata !{i32 7, i32 0, metadata !6, null}
!33 = metadata !{i32 7, i32 0, metadata !20, null}
!34 = metadata !{i32 10, i32 0, metadata !7, null}
!35 = metadata !{i32 10, i32 0, metadata !23, null}
!36 = metadata !{i32 13, i32 0, metadata !8, null}
!37 = metadata !{i32 13, i32 0, metadata !26, null}
!38 = metadata !{i32 16, i32 0, metadata !9, null}
!39 = metadata !{i32 16, i32 0, metadata !29, null}
