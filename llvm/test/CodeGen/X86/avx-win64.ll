; RUN: llc < %s -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; PR11862
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win32"

; This function has live ymm registers across a win64 call.
; The ymm6-15 registers are still call-clobbered even if xmm6-15 are callee-saved.
; Verify that callee-saved registers are not being used.

; CHECK: f___vyf
; CHECK: pushq %rbp
; CHECK: vmovmsk
; CHECK: vmovaps %ymm{{.*}}(%r
; CHECK: vmovaps %ymm{{.*}}(%r
; CHECK: call
; Two reloads. It's OK if these get folded.
; CHECK: vmovaps {{.*\(%r.*}}, %ymm
; CHECK: vmovaps {{.*\(%r.*}}, %ymm
; CHECK: blend
define <8 x float> @f___vyf(<8 x float> %x, <8 x i32> %__mask) nounwind readnone {
allocas:
  %bincmp = fcmp oeq <8 x float> %x, zeroinitializer
  %val_to_boolvec32 = sext <8 x i1> %bincmp to <8 x i32>
  %"~test" = xor <8 x i32> %val_to_boolvec32, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %"internal_mask&function_mask25" = and <8 x i32> %"~test", %__mask
  %floatmask.i46 = bitcast <8 x i32> %"internal_mask&function_mask25" to <8 x float>
  %v.i47 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i46) nounwind readnone
  %any_mm_cmp27 = icmp eq i32 %v.i47, 0
  br i1 %any_mm_cmp27, label %safe_if_after_false, label %safe_if_run_false

safe_if_run_false:                                ; preds = %allocas
  %binop = fadd <8 x float> %x, <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  %calltmp = call <8 x float> @f___vyf(<8 x float> %binop, <8 x i32> %"internal_mask&function_mask25")
  %binop33 = fadd <8 x float> %calltmp, %x
  %mask_as_float.i48 = bitcast <8 x i32> %"~test" to <8 x float>
  %blend.i52 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %x, <8 x float> %binop33, <8 x float> %mask_as_float.i48) nounwind
  br label %safe_if_after_false

safe_if_after_false:                              ; preds = %safe_if_run_false, %allocas
  %0 = phi <8 x float> [ %x, %allocas ], [ %blend.i52, %safe_if_run_false ]
  ret <8 x float> %0
}

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone
declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8*, <8 x float>) nounwind readonly
declare void @llvm.x86.avx.maskstore.ps.256(i8*, <8 x float>, <8 x float>) nounwind
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
