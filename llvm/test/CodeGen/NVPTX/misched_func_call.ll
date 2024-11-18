; RUN: llc -O3 -march=nvptx64 -enable-misched %s -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wombat(i32 %arg, i32 %arg1, i32 %arg2) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %phi = phi i32 [ 0, %bb ], [ 1, %bb3 ]
  ; CHECK:      call.uni (retval0),
  ; CHECK-NEXT: quux,
  ; CHECK-NEXT: (
  ; CHECK-NEXT: param0
  ; CHECK-NEXT: );
  %call = tail call double @quux(double 0.000000e+00)
  %mul = mul i32 %phi, %arg1
  %or = or i32 %arg2, %mul
  %mul4 = mul i32 %arg, %or
  %sitofp = sitofp i32 %mul4 to double
  %uitofp = uitofp i32 %phi to double
  %fadd = fadd double %uitofp, %sitofp
  store double %fadd, ptr addrspace(1) null, align 8
  br label %bb3
}

declare double @quux(double)
