;; __kernel void test( __global float4 *p, __global half *f )
;; {
;;   __private float4 data;
;;   data = p[0];
;;   vstorea_half4_rtp( data, 0, f );
;; }

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Float16Buffer
; CHECK-DAG: OpCapability Float16

define spir_kernel void @test(ptr addrspace(1) %p, ptr addrspace(1) %f) {
entry:
  %p.addr = alloca ptr addrspace(1), align 8
  %f.addr = alloca ptr addrspace(1), align 8
  %data = alloca <4 x float>, align 16
  store ptr addrspace(1) %p, ptr %p.addr, align 8
  store ptr addrspace(1) %f, ptr %f.addr, align 8
  %0 = load ptr addrspace(1), ptr %p.addr, align 8
  %arrayidx = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 0
  %1 = load <4 x float>, ptr addrspace(1) %arrayidx, align 16
  store <4 x float> %1, ptr %data, align 16
  %2 = load <4 x float>, ptr %data, align 16
  %3 = load ptr addrspace(1), ptr %f.addr, align 8
  call spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float> %2, i64 0, ptr addrspace(1) %3)
  ret void
}

declare spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float>, i64, ptr addrspace(1))
