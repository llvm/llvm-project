; RUN: llc -filetype=obj %s -o /dev/null
; Hanle large unsigned constant values.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-macosx10.7.0"

define zeroext i1 @_Z3iseRKxS0_(i64* nocapture %LHS, i64* nocapture %RHS) nounwind readonly optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i64* %LHS}, i64 0, metadata !7), !dbg !13
  tail call void @llvm.dbg.value(metadata !{i64* %RHS}, i64 0, metadata !11), !dbg !14
  %tmp1 = load i64* %LHS, align 4, !dbg !15, !tbaa !17
  %tmp3 = load i64* %RHS, align 4, !dbg !15, !tbaa !17
  %cmp = icmp eq i64 %tmp1, %tmp3, !dbg !15
  ret i1 %cmp, !dbg !15
}

define zeroext i1 @_Z2fnx(i64 %a) nounwind readnone optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i64 %a}, i64 0, metadata !12), !dbg !20
  tail call void @llvm.dbg.value(metadata !{i64 %a}, i64 0, metadata !12), !dbg !20
  tail call void @llvm.dbg.value(metadata !{i64 %a}, i64 0, metadata !21), !dbg !24
  tail call void @llvm.dbg.value(metadata !25, i64 0, metadata !26), !dbg !27
  %cmp.i = icmp eq i64 %a, 9223372036854775807, !dbg !28
  ret i1 %cmp.i, !dbg !22
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1, !6}
!llvm.dbg.lv._Z3iseRKxS0_ = !{!7, !11}
!llvm.dbg.lv._Z2fnx = !{!12}

!0 = metadata !{i32 655377, i32 0, i32 4, metadata !"lli.cc", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 135593)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 655406, i32 0, metadata !2, metadata !"ise", metadata !"ise", metadata !"_Z3iseRKxS0_", metadata !2, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i1 (i64*, i64*)* @_Z3iseRKxS0_, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 655401, metadata !"lli.cc", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 655381, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 655396, metadata !0, metadata !"bool", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 655406, i32 0, metadata !2, metadata !"fn", metadata !"fn", metadata !"_Z2fnx", metadata !2, i32 6, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i1 (i64)* @_Z2fnx, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 655617, metadata !1, metadata !"LHS", metadata !2, i32 16777218, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 655376, metadata !0, null, null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_reference_type ]
!9 = metadata !{i32 655398, metadata !0, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_const_type ]
!10 = metadata !{i32 655396, metadata !0, metadata !"long long int", null, i32 0, i64 64, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 655617, metadata !1, metadata !"RHS", metadata !2, i32 33554434, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 655617, metadata !6, metadata !"a", metadata !2, i32 16777222, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 2, i32 27, metadata !1, null}
!14 = metadata !{i32 2, i32 49, metadata !1, null}
!15 = metadata !{i32 3, i32 3, metadata !16, null}
!16 = metadata !{i32 655371, metadata !1, i32 2, i32 54, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!17 = metadata !{metadata !"long long", metadata !18}
!18 = metadata !{metadata !"omnipotent char", metadata !19}
!19 = metadata !{metadata !"Simple C/C++ TBAA", null}
!20 = metadata !{i32 6, i32 19, metadata !6, null}
!21 = metadata !{i32 655617, metadata !1, metadata !"LHS", metadata !2, i32 16777218, metadata !8, i32 0, metadata !22} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 7, i32 10, metadata !23, null}
!23 = metadata !{i32 655371, metadata !6, i32 6, i32 22, metadata !2, i32 1} ; [ DW_TAG_lexical_block ]
!24 = metadata !{i32 2, i32 27, metadata !1, metadata !22}
!25 = metadata !{i64 9223372036854775807}         
!26 = metadata !{i32 655617, metadata !1, metadata !"RHS", metadata !2, i32 33554434, metadata !8, i32 0, metadata !22} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 2, i32 49, metadata !1, metadata !22}
!28 = metadata !{i32 3, i32 3, metadata !16, metadata !22}
