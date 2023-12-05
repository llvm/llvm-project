;; __kernel void test( __global float4 *p, __global half *f )
;; {
;;   __private float4 data;
;;   data = p[0];
;;   vstorea_half4_rtp( data, 0, f );
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

; CHECK-SPIRV:     OpCapability Float16Buffer
; CHECK-SPIRV-NOT: OpCapability Float16

define spir_kernel void @test(<4 x float> addrspace(1)* %p, half addrspace(1)* %f) {
entry:
  %p.addr = alloca <4 x float> addrspace(1)*, align 8
  %f.addr = alloca half addrspace(1)*, align 8
  %data = alloca <4 x float>, align 16
  store <4 x float> addrspace(1)* %p, <4 x float> addrspace(1)** %p.addr, align 8
  store half addrspace(1)* %f, half addrspace(1)** %f.addr, align 8
  %0 = load <4 x float> addrspace(1)*, <4 x float> addrspace(1)** %p.addr, align 8
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %0, i64 0
  %1 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  store <4 x float> %1, <4 x float>* %data, align 16
  %2 = load <4 x float>, <4 x float>* %data, align 16
  %3 = load half addrspace(1)*, half addrspace(1)** %f.addr, align 8
  call spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float> %2, i64 0, half addrspace(1)* %3)
  ret void
}

declare spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float>, i64, half addrspace(1)*)
