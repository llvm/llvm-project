; RUN: llc < %s -mcpu=corei7-avx -mattr=+avx | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin11"

; This test would create a vpand %ymm instruction that is only legal in AVX2.
; CHECK-NOT: vpand %ymm

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone

define void @ShadeTile() nounwind {
allocas:
  br i1 undef, label %if_then, label %if_else

if_then:                                          ; preds = %allocas
  unreachable

if_else:                                          ; preds = %allocas
  br i1 undef, label %for_loop156.lr.ph, label %if_exit

for_loop156.lr.ph:                                ; preds = %if_else
  %val_6.i21244 = load i16* undef, align 2
  %0 = insertelement <8 x i16> undef, i16 %val_6.i21244, i32 6
  %val_7.i21248 = load i16* undef, align 2
  %1 = insertelement <8 x i16> %0, i16 %val_7.i21248, i32 7
  %uint2uint32.i20206 = zext <8 x i16> %1 to <8 x i32>
  %bitop5.i20208 = and <8 x i32> %uint2uint32.i20206, <i32 31744, i32 31744, i32 31744, i32 31744, i32 31744, i32 31744, i32 31744, i32 31744>
  %bitop8.i20209 = and <8 x i32> %uint2uint32.i20206, <i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023>
  %bitop12.i20211 = lshr <8 x i32> %bitop5.i20208, <i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10>
  %binop13.i20212 = add <8 x i32> %bitop12.i20211, <i32 112, i32 112, i32 112, i32 112, i32 112, i32 112, i32 112, i32 112>
  %bitop15.i20213 = shl <8 x i32> %binop13.i20212, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  %bitop17.i20214 = shl <8 x i32> %bitop8.i20209, <i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13>
  %bitop20.i20215 = or <8 x i32> undef, %bitop15.i20213
  %bitop22.i20216 = or <8 x i32> %bitop20.i20215, %bitop17.i20214
  %int_to_float_bitcast.i.i.i20217 = bitcast <8 x i32> %bitop22.i20216 to <8 x float>
  %binop401 = fmul <8 x float> undef, <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>
  %binop402 = fadd <8 x float> %binop401, <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>
  %binop403 = fmul <8 x float> zeroinitializer, %binop402
  %binop406 = fmul <8 x float> %int_to_float_bitcast.i.i.i20217, <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>
  %binop407 = fadd <8 x float> %binop406, <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>
  %binop408 = fmul <8 x float> zeroinitializer, %binop407
  %binop411 = fsub <8 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, undef
  %val_4.i21290 = load i16* undef, align 2
  %2 = insertelement <8 x i16> undef, i16 %val_4.i21290, i32 4
  %val_5.i21294 = load i16* undef, align 2
  %3 = insertelement <8 x i16> %2, i16 %val_5.i21294, i32 5
  %val_6.i21298 = load i16* undef, align 2
  %4 = insertelement <8 x i16> %3, i16 %val_6.i21298, i32 6
  %ptr_7.i21301 = inttoptr i64 undef to i16*
  %val_7.i21302 = load i16* %ptr_7.i21301, align 2
  %5 = insertelement <8 x i16> %4, i16 %val_7.i21302, i32 7
  %uint2uint32.i20218 = zext <8 x i16> %5 to <8 x i32>
  %structelement561 = load i8** undef, align 8
  %ptr2int563 = ptrtoint i8* %structelement561 to i64
  %smear.ptr_smear7571 = insertelement <8 x i64> undef, i64 %ptr2int563, i32 7
  %new_ptr582 = add <8 x i64> %smear.ptr_smear7571, zeroinitializer
  %val_5.i21509 = load i8* null, align 1
  %6 = insertelement <8 x i8> undef, i8 %val_5.i21509, i32 5
  %7 = insertelement <8 x i8> %6, i8 undef, i32 6
  %iptr_7.i21515 = extractelement <8 x i64> %new_ptr582, i32 7
  %ptr_7.i21516 = inttoptr i64 %iptr_7.i21515 to i8*
  %val_7.i21517 = load i8* %ptr_7.i21516, align 1
  %8 = insertelement <8 x i8> %7, i8 %val_7.i21517, i32 7
  %uint2float.i20245 = uitofp <8 x i8> %8 to <8 x float>
  %binop.i20246 = fmul <8 x float> %uint2float.i20245, <float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000>
  br i1 undef, label %for_loop594.lr.ph, label %for_exit595

if_exit:                                          ; preds = %if_else
  ret void

for_loop594.lr.ph:                                ; preds = %for_loop156.lr.ph
  %bitop8.i20221 = and <8 x i32> %uint2uint32.i20218, <i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023, i32 1023>
  br i1 undef, label %cif_test_all730, label %cif_mask_mixed1552

for_exit595:                                      ; preds = %for_loop156.lr.ph
  unreachable

cif_test_all730:                                  ; preds = %for_loop594.lr.ph
  %binop11.i20545 = fmul <8 x float> %binop408, zeroinitializer
  %binop12.i20546 = fadd <8 x float> undef, %binop11.i20545
  %binop15.i20547 = fmul <8 x float> %binop411, undef
  %binop16.i20548 = fadd <8 x float> %binop12.i20546, %binop15.i20547
  %bincmp774 = fcmp ogt <8 x float> %binop16.i20548, zeroinitializer
  %val_to_boolvec32775 = sext <8 x i1> %bincmp774 to <8 x i32>
  %floatmask.i20549 = bitcast <8 x i32> %val_to_boolvec32775 to <8 x float>
  %v.i20550 = tail call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i20549) nounwind readnone
  %cond = icmp eq i32 %v.i20550, 255
  br i1 %cond, label %cif_test_all794, label %cif_test_mixed

cif_test_all794:                                  ; preds = %cif_test_all730
  %binop.i20572 = fmul <8 x float> %binop403, undef
  unreachable

cif_test_mixed:                                   ; preds = %cif_test_all730
  %binop1207 = fmul <8 x float> %binop.i20246, undef
  unreachable

cif_mask_mixed1552:                               ; preds = %for_loop594.lr.ph
  unreachable
}
