; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 -verify-machineinstrs -debug-only=dagcombine -o /dev/null < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; The bitcast is used only by extracts. Collect their demands together so each
; extract is visited once, rather than once per intervening insert.
;
; CHECK-COUNT-8: Combining: {{.*}}v1f16 = extract_subvector
; CHECK-NOT: Combining: {{.*}}v1f16 = extract_subvector

target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare <1 x half> @llvm.fma.v1f16(<1 x half>, <1 x half>, <1 x half>)

define ptx_kernel void @chain(ptr addrspace(1) %in, ptr addrspace(1) %hin, ptr addrspace(1) %out) {
entry:
  %a = load <1 x half>, ptr addrspace(1) %hin, align 2
  %hb = getelementptr half, ptr addrspace(1) %hin, i64 1
  %b = load <1 x half>, ptr addrspace(1) %hb, align 2
  %x0 = load i32, ptr addrspace(1) %in, align 4
  %p1 = getelementptr i32, ptr addrspace(1) %in, i64 1
  %x1 = load i32, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr i32, ptr addrspace(1) %in, i64 2
  %x2 = load i32, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr i32, ptr addrspace(1) %in, i64 3
  %x3 = load i32, ptr addrspace(1) %p3, align 4
  %v0 = insertelement <4 x i32> poison, i32 %x0, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 %x1, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 %x2, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 %x3, i64 3
  %halves = bitcast <4 x i32> %v3 to <8 x half>
  %e0 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 0>
  %f0 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e0, <1 x half> %a, <1 x half> %b)
  %e1 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 1>
  %f1 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e1, <1 x half> %a, <1 x half> %b)
  %e2 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 2>
  %f2 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e2, <1 x half> %a, <1 x half> %b)
  %e3 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 3>
  %f3 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e3, <1 x half> %a, <1 x half> %b)
  %e4 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 4>
  %f4 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e4, <1 x half> %a, <1 x half> %b)
  %e5 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 5>
  %f5 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e5, <1 x half> %a, <1 x half> %b)
  %e6 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 6>
  %f6 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e6, <1 x half> %a, <1 x half> %b)
  %e7 = shufflevector <8 x half> %halves, <8 x half> poison, <1 x i32> <i32 7>
  %f7 = call <1 x half> @llvm.fma.v1f16(<1 x half> %e7, <1 x half> %a, <1 x half> %b)
  %q0 = shufflevector <1 x half> %f0, <1 x half> %f1, <2 x i32> <i32 0, i32 1>
  %r0 = bitcast <2 x half> %q0 to i32
  %q1 = shufflevector <1 x half> %f2, <1 x half> %f3, <2 x i32> <i32 0, i32 1>
  %r1 = bitcast <2 x half> %q1 to i32
  %q2 = shufflevector <1 x half> %f4, <1 x half> %f5, <2 x i32> <i32 0, i32 1>
  %r2 = bitcast <2 x half> %q2 to i32
  %q3 = shufflevector <1 x half> %f6, <1 x half> %f7, <2 x i32> <i32 0, i32 1>
  %r3 = bitcast <2 x half> %q3 to i32
  store i32 %r0, ptr addrspace(1) %out, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %r1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i64 2
  store i32 %r2, ptr addrspace(1) %out2, align 4
  %out3 = getelementptr i32, ptr addrspace(1) %out, i64 3
  store i32 %r3, ptr addrspace(1) %out3, align 4
  ret void
}
