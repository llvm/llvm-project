; RUN: llc -O3 -march=nvptx64 -enable-misched %s -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @my_kernel(i32 %arg_0, i32 %arg_3.tr, i32 %"$$i_l40_0_t23.0") {
Entry_BB:
  br label %BB1692

BB1692:                                           ; preds = %BB1692, %Entry_BB
  %"$$i_l40_0_t23.02" = phi i32 [ 0, %Entry_BB ], [ 1, %BB1692 ]
  ; CHECK:      call.uni (retval0),
  ; CHECK-NEXT: _FOO,
  ; CHECK-NEXT: (
  ; CHECK-NEXT: param0
  ; CHECK-NEXT: );
  %r55 = tail call double @_FOO(double 0.000000e+00)
  %0 = mul i32 %"$$i_l40_0_t23.02", %arg_3.tr
  %1 = or i32 %"$$i_l40_0_t23.0", %0
  %r59 = mul i32 %arg_0, %1
  %r61 = sitofp i32 %r59 to double
  %r66 = uitofp i32 %"$$i_l40_0_t23.02" to double
  %r68 = fadd double %r66, %r61
  store double %r68, ptr addrspace(1) null, align 8
  br label %BB1692
}

declare double @_FOO(double)
