; RUN: opt -S -deadargelim < %s | FileCheck %s

@.str = private constant [1 x i8] zeroinitializer, align 1 ; <[1 x i8]*> [#uses=1]

define i8* @vfs_addname(i8* %name, i32 %len, i32 %hash, i32 %flags) nounwind ssp {
entry:
  call void @llvm.dbg.value(metadata i8* %name, i64 0, metadata !0, metadata !{})
  call void @llvm.dbg.value(metadata i32 %len, i64 0, metadata !10, metadata !{})
  call void @llvm.dbg.value(metadata i32 %hash, i64 0, metadata !11, metadata !{})
  call void @llvm.dbg.value(metadata i32 %flags, i64 0, metadata !12, metadata !{})
; CHECK:  call fastcc i8* @add_name_internal(i8* %name, i32 %hash) [[NUW:#[0-9]+]], !dbg !{{[0-9]+}}
  %0 = call fastcc i8* @add_name_internal(i8* %name, i32 %len, i32 %hash, i8 zeroext 0, i32 %flags) nounwind, !dbg !13 ; <i8*> [#uses=1]
  ret i8* %0, !dbg !13
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define internal fastcc i8* @add_name_internal(i8* %name, i32 %len, i32 %hash, i8 zeroext %extra, i32 %flags) noinline nounwind ssp {
entry:
  call void @llvm.dbg.value(metadata i8* %name, i64 0, metadata !15, metadata !{})
  call void @llvm.dbg.value(metadata i32 %len, i64 0, metadata !20, metadata !{})
  call void @llvm.dbg.value(metadata i32 %hash, i64 0, metadata !21, metadata !{})
  call void @llvm.dbg.value(metadata i8 %extra, i64 0, metadata !22, metadata !{})
  call void @llvm.dbg.value(metadata i32 %flags, i64 0, metadata !23, metadata !{})
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

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

; CHECK: attributes #0 = { nounwind ssp }
; CHECK: attributes #1 = { nounwind readnone }
; CHECK: attributes #2 = { noinline nounwind ssp }
; CHECK: attributes [[NUW]] = { nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}
!0 = !{!"0x101\00name\008\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00vfs_addname\00vfs_addname\00vfs_addname\0012\000\001\000\006\000\000\000", !28, !2, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !28} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)\001\00\000\00\000", !28, !29, !29, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !28, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !6, !9, !9, !9}
!6 = !{!"0xf\00\000\0064\0064\000\000", !28, !2, !7} ; [ DW_TAG_pointer_type ]
!7 = !{!"0x26\00\000\008\008\000\000", !28, !2, !8} ; [ DW_TAG_const_type ]
!8 = !{!"0x24\00char\000\008\008\000\000\006", !28, !2} ; [ DW_TAG_base_type ]
!9 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", !28, !2} ; [ DW_TAG_base_type ]
!10 = !{!"0x101\00len\009\000", !1, !2, !9} ; [ DW_TAG_arg_variable ]
!11 = !{!"0x101\00hash\0010\000", !1, !2, !9} ; [ DW_TAG_arg_variable ]
!12 = !{!"0x101\00flags\0011\000", !1, !2, !9} ; [ DW_TAG_arg_variable ]
!13 = !MDLocation(line: 13, scope: !14)
!14 = !{!"0xb\0012\000\000", !28, !1} ; [ DW_TAG_lexical_block ]
!15 = !{!"0x101\00name\0017\000", !16, !2, !6} ; [ DW_TAG_arg_variable ]
!16 = !{!"0x2e\00add_name_internal\00add_name_internal\00add_name_internal\0022\001\001\000\006\000\000\000", !28, !2, !17, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!17 = !{!"0x15\00\000\000\000\000\000\000", !28, !2, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{!6, !6, !9, !9, !19, !9}
!19 = !{!"0x24\00unsigned char\000\008\008\000\000\008", !28, !2} ; [ DW_TAG_base_type ]
!20 = !{!"0x101\00len\0018\000", !16, !2, !9} ; [ DW_TAG_arg_variable ]
!21 = !{!"0x101\00hash\0019\000", !16, !2, !9} ; [ DW_TAG_arg_variable ]
!22 = !{!"0x101\00extra\0020\000", !16, !2, !19} ; [ DW_TAG_arg_variable ]
!23 = !{!"0x101\00flags\0021\000", !16, !2, !9} ; [ DW_TAG_arg_variable ]
!24 = !MDLocation(line: 23, scope: !25)
!25 = !{!"0xb\0022\000\000", !28, !16} ; [ DW_TAG_lexical_block ]
!26 = !MDLocation(line: 24, scope: !25)
!27 = !MDLocation(line: 26, scope: !25)
!28 = !{!"tail.c", !"/Users/echeng/LLVM/radars/r7927803/"}
!29 = !{i32 0}
!30 = !{i32 1, !"Debug Info Version", i32 2}
