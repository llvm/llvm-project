; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: 	vadd.f32	q4, q8, q8
;CHECK-NEXT: LBB0_1

;CHECK:@DEBUG_VALUE: x <- Q4{{$}}
;CHECK-NEXT:@DEBUG_VALUE: y <- Q4{{$}}


@.str = external constant [13 x i8]

declare <4 x float> @test0001(float) nounwind readnone ssp

define i32 @main(i32 %argc, i8** nocapture %argv, i1 %cond) nounwind ssp {
entry:
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %add19 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  tail call void @llvm.dbg.value(metadata !{<4 x float> %add19}, i64 0, metadata !27), !dbg !39
  %add20 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  tail call void @llvm.dbg.value(metadata !{<4 x float> %add20}, i64 0, metadata !28), !dbg !39
  br i1 %cond, label %for.end54, label %for.body9, !dbg !44

for.end54:                                        ; preds = %for.body9
  %tmp115 = extractelement <4 x float> %add19, i32 1
  %conv6.i75 = fpext float %tmp115 to double, !dbg !45
  %call.i82 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i75, double undef, double undef) nounwind, !dbg !45
  %tmp116 = extractelement <4 x float> %add20, i32 1
  %conv6.i76 = fpext float %tmp116 to double, !dbg !45
  %call.i83 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i76, double undef, double undef) nounwind, !dbg !45
  ret i32 0, !dbg !49
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.module.flags = !{!56}

!0 = metadata !{i32 786478, metadata !54, null, metadata !"test0001", metadata !"test0001", metadata !"", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, <4 x float> (float)* @test0001, null, null, metadata !51, i32 0} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !54} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !54, i32 12, metadata !"clang version 3.0 (trunk 129915)", i1 true, metadata !"", i32 0, metadata !17, metadata !17, metadata !50, null,  null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !54, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786454, metadata !54, metadata !2, metadata !"v4f32", i32 14, i64 0, i64 0, i64 0, i32 0, metadata !6} ; [ DW_TAG_typedef ]
!6 = metadata !{i32 786433, metadata !54, metadata !2, metadata !"", i32 0, i64 128, i64 128, i32 0, i32 0, metadata !7, metadata !8, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [from float]
!7 = metadata !{i32 786468, null, metadata !2, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786465, i64 0, i64 4}         ; [ DW_TAG_subrange_type ]
!10 = metadata !{i32 786478, metadata !54, null, metadata !"main", metadata !"main", metadata !"", i32 59, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i8**, i1)* @main, null, null, metadata !52, i32 0} ; [ DW_TAG_subprogram ] [line 59] [def] [scope 0] [main]
!11 = metadata !{i32 786453, metadata !54, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786468, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 786478, metadata !55, null, metadata !"printFV", metadata !"printFV", metadata !"", i32 41, metadata !16, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, null, metadata !53, i32 0} ; [ DW_TAG_subprogram ] [line 41] [local] [def] [scope 0] [printFV]
!15 = metadata !{i32 786473, metadata !55} ; [ DW_TAG_file_type ]
!16 = metadata !{i32 786453, metadata !55, metadata !15, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null}
!18 = metadata !{i32 786689, metadata !0, metadata !"a", metadata !1, i32 16777219, metadata !7, i32 0, null} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 786689, metadata !10, metadata !"argc", metadata !1, i32 16777275, metadata !13, i32 0, null} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 786689, metadata !10, metadata !"argv", metadata !1, i32 33554491, metadata !21, i32 0, null} ; [ DW_TAG_arg_variable ]
!21 = metadata !{i32 786447, null, metadata !2, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !22} ; [ DW_TAG_pointer_type ]
!22 = metadata !{i32 786447, null, metadata !2, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{i32 786468, null, metadata !2, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!24 = metadata !{i32 786688, metadata !25, metadata !"i", metadata !1, i32 60, metadata !13, i32 0, null} ; [ DW_TAG_auto_variable ]
!25 = metadata !{i32 786443, metadata !1, metadata !10, i32 59, i32 33, i32 14} ; [ DW_TAG_lexical_block ]
!26 = metadata !{i32 786688, metadata !25, metadata !"j", metadata !1, i32 60, metadata !13, i32 0, null} ; [ DW_TAG_auto_variable ]
!27 = metadata !{i32 786688, metadata !25, metadata !"x", metadata !1, i32 61, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!28 = metadata !{i32 786688, metadata !25, metadata !"y", metadata !1, i32 62, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 786688, metadata !25, metadata !"z", metadata !1, i32 63, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!30 = metadata !{i32 786689, metadata !14, metadata !"F", metadata !15, i32 16777257, metadata !31, i32 0, null} ; [ DW_TAG_arg_variable ]
!31 = metadata !{i32 786447, null, metadata !2, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_pointer_type ]
!32 = metadata !{i32 786454, metadata !55, metadata !2, metadata !"FV", i32 25, i64 0, i64 0, i64 0, i32 0, metadata !33} ; [ DW_TAG_typedef ]
!33 = metadata !{i32 786455, metadata !55, metadata !2, metadata !"", i32 22, i64 128, i64 128, i64 0, i32 0, i32 0, metadata !34, i32 0, i32 0} ; [ DW_TAG_union_type ]
!34 = metadata !{metadata !35, metadata !37}
!35 = metadata !{i32 786445, metadata !55, metadata !15, metadata !"V", i32 23, i64 128, i64 128, i64 0, i32 0, metadata !36} ; [ DW_TAG_member ]
!36 = metadata !{i32 786454, metadata !55, metadata !2, metadata !"v4sf", i32 3, i64 0, i64 0, i64 0, i32 0, metadata !6} ; [ DW_TAG_typedef ]
!37 = metadata !{i32 786445, metadata !55, metadata !15, metadata !"A", i32 24, i64 128, i64 32, i64 0, i32 0, metadata !38} ; [ DW_TAG_member ]
!38 = metadata !{i32 786433, null, metadata !2, metadata !"", i32 0, i64 128, i64 32, i32 0, i32 0, metadata !7, metadata !8, i32 0, i32 0} ; [ DW_TAG_array_type ]
!39 = metadata !{i32 79, i32 7, metadata !40, null}
!40 = metadata !{i32 786443, metadata !1, metadata !41, i32 75, i32 35, i32 18} ; [ DW_TAG_lexical_block ]
!41 = metadata !{i32 786443, metadata !1, metadata !42, i32 75, i32 5, i32 17} ; [ DW_TAG_lexical_block ]
!42 = metadata !{i32 786443, metadata !1, metadata !43, i32 71, i32 32, i32 16} ; [ DW_TAG_lexical_block ]
!43 = metadata !{i32 786443, metadata !1, metadata !25, i32 71, i32 3, i32 15} ; [ DW_TAG_lexical_block ]
!44 = metadata !{i32 75, i32 5, metadata !42, null}
!45 = metadata !{i32 42, i32 2, metadata !46, metadata !48}
!46 = metadata !{i32 786443, metadata !15, metadata !47, i32 42, i32 2, i32 20} ; [ DW_TAG_lexical_block ]
!47 = metadata !{i32 786443, metadata !15, metadata !14, i32 41, i32 28, i32 19} ; [ DW_TAG_lexical_block ]
!48 = metadata !{i32 95, i32 3, metadata !25, null}
!49 = metadata !{i32 99, i32 3, metadata !25, null}
!50 = metadata !{metadata !0, metadata !10, metadata !14}
!51 = metadata !{metadata !18}
!52 = metadata !{metadata !19, metadata !20, metadata !24, metadata !26, metadata !27, metadata !28, metadata !29}
!53 = metadata !{metadata !30}
!54 = metadata !{metadata !"build2.c", metadata !"/private/tmp"}
!55 = metadata !{metadata !"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", metadata !"/private/tmp"}
!56 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
