; RUN: opt < %s -verify-ipgo -passes=loop-unroll -S -disable-output 2>&1 | FileCheck %s

; CHECK: *** IPGO Verification After LoopUnrollPass ***
; CHECK-NEXT: PGOVerify# Block frequency mismatch in function lzma_vli_size, block do.body.peel:  Incoming=75:  Outgoing=163
; CHECK-NEXT: PGOVerify# Block frequency mismatch in function lzma_vli_size, block do.body.peel2:  Incoming=88:  Outgoing=163
; CHECK-NEXT: PGOVerify# Block frequency mismatch in function lzma_vli_size, block do.body:  Incoming=176:  Outgoing=163

; ModuleID = 'blockfrequency.ll'
source_filename = "vli_size.c"

; Function Attrs: cold nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @lzma_vli_size(i64 noundef %vli) local_unnamed_addr #0 !prof !34 {
entry:
  %cmp = icmp slt i64 %vli, 0
  br i1 %cmp, label %return, label %do.body.preheader, !prof !35
do.body.preheader:                                ; preds = %entry
  br label %do.body
do.body:                                          ; preds = %do.body.preheader, %do.body
  %vli.addr.0 = phi i64 [ %shr, %do.body ], [ %vli, %do.body.preheader ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %do.body.preheader ]
  %inc = add nuw nsw i32 %i.0, 1
  %cmp1.not = icmp samesign ult i64 %vli.addr.0, 128
  %shr = lshr i64 %vli.addr.0, 7
  br i1 %cmp1.not, label %return.loopexit, label %do.body, !prof !36, !llvm.loop !37
return.loopexit:                                  ; preds = %do.body
  %inc.lcssa = phi i32 [ %inc, %do.body ]
  br label %return
return:                                           ; preds = %return.loopexit, %entry
  %retval.0 = phi i32 [ 0, %entry ], [ %inc.lcssa, %return.loopexit ]
  ret i32 %retval.0
}
attributes #0 = { cold nofree norecurse nosync nounwind memory(none) uwtable "approx-func-fp-math"="true" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "t
arget-cpu"="znver4" "target-features"="+adx,+aes,+avx,+avx2,+avx512bf16,+avx512\0Abitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+
fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+pku,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="true" }
!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!33}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!4 = !{i32 1, !"ProfileSummary", !5}
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!6 = !{!"ProfileFormat", !"InstrProf"}
!7 = !{!"TotalCount", i64 162836809056}
!8 = !{!"MaxCount", i64 43208067370}
!9 = !{!"MaxInternalCount", i64 43208067370}
!10 = !{!"MaxFunctionCount", i64 4237001992}
!11 = !{!"NumCounts", i64 2798}
!12 = !{!"NumFunctions", i64 380}
!13 = !{!"IsPartialProfile", i64 0}
!14 = !{!"PartialProfileRatio", double 0.000000e+00}
!15 = !{!"DetailedSummary", !16}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}
!17 = !{i32 10000, i64 43208067370, i32 1}
!18 = !{i32 100000, i64 43208067370, i32 1}
!19 = !{i32 200000, i64 43208067370, i32 1}
!20 = !{i32 300000, i64 23950887848, i32 2}
!21 = !{i32 400000, i64 23950887848, i32 2}
!22 = !{i32 500000, i64 7507132423, i32 4}
!23 = !{i32 600000, i64 4237001992, i32 7}
!24 = !{i32 700000, i64 1929299908, i32 13}
!25 = !{i32 800000, i64 1033074021, i32 23}
!26 = !{i32 900000, i64 481961878, i32 47}
!27 = !{i32 950000, i64 155061509, i32 77}
!28 = !{i32 990000, i64 21722093, i32 157}
!29 = !{i32 999000, i64 2514526, i32 324}
!30 = !{i32 999900, i64 243879, i32 447}
!31 = !{i32 999990, i64 84576, i32 542}
!32 = !{i32 999999, i64 7980, i32 576}
!33 = !{!"AMD clang version 21.1.8pre (CLANG: AOCC_6.0.0pre-Build#4-gf926354d042f 2026_03_26 Prerelease)"}
!34 = !{!"function_entry_count", i64 75}
!35 = !{!"branch_weights", i32 0, i32 75}
!36 = !{!"branch_weights", i32 75, i32 88}
!37 = distinct !{!37, !38}
!38 = !{!"llvm.loop.mustprogress"}
