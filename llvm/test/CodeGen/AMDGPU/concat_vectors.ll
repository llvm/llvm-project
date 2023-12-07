; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}test_concat_v1i32:
; 0x80f000 is the high 32 bits of the resource descriptor used by MUBUF
; instructions that access scratch memory.  Bit 23, which is the add_tid_enable
; bit, is only set for scratch access, so we can check for the absence of this
; value if we want to ensure scratch memory is not being used.
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1i32(ptr addrspace(1) %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %concat = shufflevector <1 x i32> %a, <1 x i32> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i32> %concat, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2i32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2i32(ptr addrspace(1) %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %concat = shufflevector <2 x i32> %a, <2 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i32> %concat, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4i32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4i32(ptr addrspace(1) %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %concat = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %concat, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8i32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8i32(ptr addrspace(1) %out, <8 x i32> %a, <8 x i32> %b) nounwind {
  %concat = shufflevector <8 x i32> %a, <8 x i32> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i32> %concat, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16i32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16i32(ptr addrspace(1) %out, <16 x i32> %a, <16 x i32> %b) nounwind {
  %concat = shufflevector <16 x i32> %a, <16 x i32> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %concat, ptr addrspace(1) %out, align 128
  ret void
}

; GCN-LABEL: {{^}}test_concat_v1f32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1f32(ptr addrspace(1) %out, <1 x float> %a, <1 x float> %b) nounwind {
  %concat = shufflevector <1 x float> %a, <1 x float> %b, <2 x i32> <i32 0, i32 1>
  store <2 x float> %concat, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2f32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2f32(ptr addrspace(1) %out, <2 x float> %a, <2 x float> %b) nounwind {
  %concat = shufflevector <2 x float> %a, <2 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x float> %concat, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4f32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4f32(ptr addrspace(1) %out, <4 x float> %a, <4 x float> %b) nounwind {
  %concat = shufflevector <4 x float> %a, <4 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x float> %concat, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8f32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8f32(ptr addrspace(1) %out, <8 x float> %a, <8 x float> %b) nounwind {
  %concat = shufflevector <8 x float> %a, <8 x float> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x float> %concat, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16f32:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16f32(ptr addrspace(1) %out, <16 x float> %a, <16 x float> %b) nounwind {
  %concat = shufflevector <16 x float> %a, <16 x float> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x float> %concat, ptr addrspace(1) %out, align 128
  ret void
}

; GCN-LABEL: {{^}}test_concat_v1i64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1i64(ptr addrspace(1) %out, <1 x double> %a, <1 x double> %b) nounwind {
  %concat = shufflevector <1 x double> %a, <1 x double> %b, <2 x i32> <i32 0, i32 1>
  store <2 x double> %concat, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2i64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2i64(ptr addrspace(1) %out, <2 x double> %a, <2 x double> %b) nounwind {
  %concat = shufflevector <2 x double> %a, <2 x double> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x double> %concat, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4i64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4i64(ptr addrspace(1) %out, <4 x double> %a, <4 x double> %b) nounwind {
  %concat = shufflevector <4 x double> %a, <4 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x double> %concat, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8i64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8i64(ptr addrspace(1) %out, <8 x double> %a, <8 x double> %b) nounwind {
  %concat = shufflevector <8 x double> %a, <8 x double> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x double> %concat, ptr addrspace(1) %out, align 128
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16i64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16i64(ptr addrspace(1) %out, <16 x double> %a, <16 x double> %b) nounwind {
  %concat = shufflevector <16 x double> %a, <16 x double> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x double> %concat, ptr addrspace(1) %out, align 256
  ret void
}

; GCN-LABEL: {{^}}test_concat_v1f64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1f64(ptr addrspace(1) %out, <1 x double> %a, <1 x double> %b) nounwind {
  %concat = shufflevector <1 x double> %a, <1 x double> %b, <2 x i32> <i32 0, i32 1>
  store <2 x double> %concat, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2f64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2f64(ptr addrspace(1) %out, <2 x double> %a, <2 x double> %b) nounwind {
  %concat = shufflevector <2 x double> %a, <2 x double> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x double> %concat, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4f64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4f64(ptr addrspace(1) %out, <4 x double> %a, <4 x double> %b) nounwind {
  %concat = shufflevector <4 x double> %a, <4 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x double> %concat, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8f64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8f64(ptr addrspace(1) %out, <8 x double> %a, <8 x double> %b) nounwind {
  %concat = shufflevector <8 x double> %a, <8 x double> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x double> %concat, ptr addrspace(1) %out, align 128
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16f64:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16f64(ptr addrspace(1) %out, <16 x double> %a, <16 x double> %b) nounwind {
  %concat = shufflevector <16 x double> %a, <16 x double> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x double> %concat, ptr addrspace(1) %out, align 256
  ret void
}

; GCN-LABEL: {{^}}test_concat_v1i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1i1(ptr addrspace(1) %out, <1 x i1> %a, <1 x i1> %b) nounwind {
  %concat = shufflevector <1 x i1> %a, <1 x i1> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2i1(ptr addrspace(1) %out, <2 x i1> %a, <2 x i1> %b) nounwind {
  %concat = shufflevector <2 x i1> %a, <2 x i1> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4i1(ptr addrspace(1) %out, <4 x i1> %a, <4 x i1> %b) nounwind {
  %concat = shufflevector <4 x i1> %a, <4 x i1> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8i1(ptr addrspace(1) %out, <8 x i1> %a, <8 x i1> %b) nounwind {
  %concat = shufflevector <8 x i1> %a, <8 x i1> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16i1(ptr addrspace(1) %out, <16 x i1> %a, <16 x i1> %b) nounwind {
  %concat = shufflevector <16 x i1> %a, <16 x i1> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v32i1:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v32i1(ptr addrspace(1) %out, <32 x i1> %a, <32 x i1> %b) nounwind {
  %concat = shufflevector <32 x i1> %a, <32 x i1> %b, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x i1> %concat, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_concat_v1i16:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v1i16(ptr addrspace(1) %out, <1 x i16> %a, <1 x i16> %b) nounwind {
  %concat = shufflevector <1 x i16> %a, <1 x i16> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i16> %concat, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_concat_v2i16:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v2i16(ptr addrspace(1) %out, <2 x i16> %a, <2 x i16> %b) nounwind {
  %concat = shufflevector <2 x i16> %a, <2 x i16> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %concat, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_concat_v4i16:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v4i16(ptr addrspace(1) %out, <4 x i16> %a, <4 x i16> %b) nounwind {
  %concat = shufflevector <4 x i16> %a, <4 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i16> %concat, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_concat_v8i16:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v8i16(ptr addrspace(1) %out, <8 x i16> %a, <8 x i16> %b) nounwind {
  %concat = shufflevector <8 x i16> %a, <8 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i16> %concat, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_concat_v16i16:
; GCN-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; GCN-NOT: movrel
define amdgpu_kernel void @test_concat_v16i16(ptr addrspace(1) %out, <16 x i16> %a, <16 x i16> %b) nounwind {
  %concat = shufflevector <16 x i16> %a, <16 x i16> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i16> %concat, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}concat_vector_crash:
; GCN: s_endpgm
define amdgpu_kernel void @concat_vector_crash(ptr addrspace(1) %out, ptr addrspace(1) %in) {
bb:
  %tmp = load <2 x float>, ptr addrspace(1) %in, align 4
  %tmp1 = shufflevector <2 x float> %tmp, <2 x float> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %tmp2 = shufflevector <8 x float> undef, <8 x float> %tmp1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  store <8 x float> %tmp2, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}concat_vector_crash2:
; GCN: s_endpgm
define amdgpu_kernel void @concat_vector_crash2(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %tmp = load i32, ptr addrspace(1) %in, align 1
  %tmp1 = trunc i32 %tmp to i24
  %tmp2 = bitcast i24 %tmp1 to <3 x i8>
  %tmp3 = shufflevector <3 x i8> %tmp2, <3 x i8> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 1, i32 undef, i32 undef>
  %tmp4 = shufflevector <8 x i8> %tmp3, <8 x i8> <i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 7, i8 8>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 14, i32 15>
  store <8 x i8> %tmp4, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}build_vector_splat_concat_v8i16:
; VI: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; VI: ds_write_b128
; VI: ds_write_b128
define amdgpu_kernel void @build_vector_splat_concat_v8i16() {
entry:
  store <8 x i16> zeroinitializer, ptr addrspace(3) undef, align 16
  store <8 x i16> zeroinitializer, ptr addrspace(3) null, align 16
  ret void
}
