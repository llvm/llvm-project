; RUN: opt -S -deadargelim < %s | FileCheck %s

@.str = private constant [1 x i8] zeroinitializer, align 1 ; <[1 x i8]*> [#uses=1]

define i8* @vfs_addname(i8* %name, i32 %len, i32 %hash, i32 %flags) nounwind ssp {
entry:
  call void @llvm.dbg.value(metadata !{i8* %name}, i64 0, metadata !0)
  call void @llvm.dbg.value(metadata !{i32 %len}, i64 0, metadata !10)
  call void @llvm.dbg.value(metadata !{i32 %hash}, i64 0, metadata !11)
  call void @llvm.dbg.value(metadata !{i32 %flags}, i64 0, metadata !12)
; CHECK:  call fastcc i8* @add_name_internal(i8* %name, i32 %hash) nounwind, !dbg !13
  %0 = call fastcc i8* @add_name_internal(i8* %name, i32 %len, i32 %hash, i8 zeroext 0, i32 %flags) nounwind, !dbg !13 ; <i8*> [#uses=1]
  ret i8* %0, !dbg !13
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define internal fastcc i8* @add_name_internal(i8* %name, i32 %len, i32 %hash, i8 zeroext %extra, i32 %flags) nounwind noinline ssp {
entry:
  call void @llvm.dbg.value(metadata !{i8* %name}, i64 0, metadata !15)
  call void @llvm.dbg.value(metadata !{i32 %len}, i64 0, metadata !20)
  call void @llvm.dbg.value(metadata !{i32 %hash}, i64 0, metadata !21)
  call void @llvm.dbg.value(metadata !{i8 %extra}, i64 0, metadata !22)
  call void @llvm.dbg.value(metadata !{i32 %flags}, i64 0, metadata !23)
  %0 = icmp eq i32 %hash, 0, !dbg !24             ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1, !dbg !24

bb:                                               ; preds = %entry
  br label %bb2, !dbg !26

bb1:                                              ; preds = %entry
  br label %bb2, !dbg !27

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i8* [ getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), %bb ], [ %name, %bb1 ] ; <i8*> [#uses=1]
  ret i8* %.0, !dbg !27
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!0 = metadata !{i32 524545, metadata !1, metadata !"name", metadata !2, i32 8, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"vfs_addname", metadata !"vfs_addname", metadata !"vfs_addname", metadata !2, i32 12, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"tail.c", metadata !"/Users/echeng/LLVM/radars/r7927803/", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"tail.c", metadata !"/Users/echeng/LLVM/radars/r7927803/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6, metadata !6, metadata !9, metadata !9, metadata !9}
!6 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 524326, metadata !2, metadata !"", metadata !2, i32 0, i64 8, i64 8, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ]
!8 = metadata !{i32 524324, metadata !2, metadata !"char", metadata !2, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524324, metadata !2, metadata !"unsigned int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 524545, metadata !1, metadata !"len", metadata !2, i32 9, metadata !9} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 524545, metadata !1, metadata !"hash", metadata !2, i32 10, metadata !9} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 524545, metadata !1, metadata !"flags", metadata !2, i32 11, metadata !9} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 13, i32 0, metadata !14, null}
!14 = metadata !{i32 524299, metadata !1, i32 12, i32 0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 524545, metadata !16, metadata !"name", metadata !2, i32 17, metadata !6} ; [ DW_TAG_arg_variable ]
!16 = metadata !{i32 524334, i32 0, metadata !2, metadata !"add_name_internal", metadata !"add_name_internal", metadata !"add_name_internal", metadata !2, i32 22, metadata !17, i1 true, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null} ; [ DW_TAG_subroutine_type ]
!18 = metadata !{metadata !6, metadata !6, metadata !9, metadata !9, metadata !19, metadata !9}
!19 = metadata !{i32 524324, metadata !2, metadata !"unsigned char", metadata !2, i32 0, i64 8, i64 8, i64 0, i32 0, i32 8} ; [ DW_TAG_base_type ]
!20 = metadata !{i32 524545, metadata !16, metadata !"len", metadata !2, i32 18, metadata !9} ; [ DW_TAG_arg_variable ]
!21 = metadata !{i32 524545, metadata !16, metadata !"hash", metadata !2, i32 19, metadata !9} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 524545, metadata !16, metadata !"extra", metadata !2, i32 20, metadata !19} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 524545, metadata !16, metadata !"flags", metadata !2, i32 21, metadata !9} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 23, i32 0, metadata !25, null}
!25 = metadata !{i32 524299, metadata !16, i32 22, i32 0} ; [ DW_TAG_lexical_block ]
!26 = metadata !{i32 24, i32 0, metadata !25, null}
!27 = metadata !{i32 26, i32 0, metadata !25, null}
