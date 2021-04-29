; RUN: llc -aarch64-sve-vector-bits-min=256 < %s | FileCheck %s

target triple = "aarch64"

;
; NOTE: SVE lowering for the BSP pseudoinst is not currently implemented, so we
;       don't currently expect the code below to lower to BSL/BIT/BIF. Once
;       this is implemented, this test will be fleshed out.
;

define <8 x i32> @fixed_bitselect_v8i32(<8 x i32>* %pre_cond_ptr, <8 x i32>* %left_ptr, <8 x i32>* %right_ptr) #0 {
; CHECK-LABEL: fixed_bitselect_v8i32:
; CHECK-NOT:     bsl {{.*}}, {{.*}}, {{.*}}
; CHECK-NOT:     bit {{.*}}, {{.*}}, {{.*}}
; CHECK-NOT:     bif {{.*}}, {{.*}}, {{.*}}
; CHECK:         ret
  %pre_cond = load <8 x i32>, <8 x i32>* %pre_cond_ptr
  %left = load <8 x i32>, <8 x i32>* %left_ptr
  %right = load <8 x i32>, <8 x i32>* %right_ptr

  %neg_cond = sub <8 x i32> zeroinitializer, %pre_cond
  %min_cond = add <8 x i32> %pre_cond, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %left_bits_0 = and <8 x i32> %neg_cond, %left
  %right_bits_0 = and <8 x i32> %min_cond, %right
  %bsl0000 = or <8 x i32> %right_bits_0, %left_bits_0
  ret <8 x i32> %bsl0000
}

attributes #0 = { "target-features"="+sve" }
