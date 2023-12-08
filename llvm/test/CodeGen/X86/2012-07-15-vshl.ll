; RUN: llc < %s -mtriple=i686-- -mcpu=corei7 -mattr=+avx
; PR13352

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define void @f_f() nounwind {
allocas:
  br label %for_loop29

for_loop29:                                       ; preds = %safe_if_after_true, %allocas
  %indvars.iv596 = phi i64 [ %indvars.iv.next597, %safe_if_after_true ], [ 0, %allocas ]
  %0 = trunc i64 %indvars.iv596 to i32
  %smear.15 = insertelement <16 x i32> undef, i32 %0, i32 15
  %bitop = lshr <16 x i32> zeroinitializer, %smear.15
  %bitop35 = and <16 x i32> %bitop, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %bitop35_to_bool = icmp ne <16 x i32> %bitop35, zeroinitializer
  %val_to_boolvec32 = sext <16 x i1> %bitop35_to_bool to <16 x i32>
  %floatmask.i526 = bitcast <16 x i32> %val_to_boolvec32 to <16 x float>
  %mask1.i529 = shufflevector <16 x float> %floatmask.i526, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %"internal_mask&function_mask41_any" = icmp eq i32 undef, 0
  br i1 %"internal_mask&function_mask41_any", label %safe_if_after_true, label %safe_if_run_true

safe_if_after_true:                               ; preds = %for_loop29
  %indvars.iv.next597 = add i64 %indvars.iv596, 1
  br label %for_loop29

safe_if_run_true:                                 ; preds = %for_loop29
  %blend1.i583 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> undef, <8 x float> undef, <8 x float> %mask1.i529) nounwind
  unreachable
}

