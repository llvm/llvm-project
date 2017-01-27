; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_concat_v1i32:
; 0x80f000 is the high 32 bits of the resource descriptor used by MUBUF
; instructions that access scratch memory.  Bit 23, which is the add_tid_enable
; bit, is only set for scratch access, so we can check for the absence of this
; value if we want to ensure scratch memory is not being used.
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1i32(<2 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %concat = shufflevector <1 x i32> %a, <1 x i32> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i32> %concat, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2i32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2i32(<4 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %concat = shufflevector <2 x i32> %a, <2 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i32> %concat, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4i32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4i32(<8 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %concat = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %concat, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8i32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8i32(<16 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) nounwind {
  %concat = shufflevector <8 x i32> %a, <8 x i32> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i32> %concat, <16 x i32> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16i32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16i32(<32 x i32> addrspace(1)* %out, <16 x i32> %a, <16 x i32> %b) nounwind {
  %concat = shufflevector <16 x i32> %a, <16 x i32> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %concat, <32 x i32> addrspace(1)* %out, align 128
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v1f32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1f32(<2 x float> addrspace(1)* %out, <1 x float> %a, <1 x float> %b) nounwind {
  %concat = shufflevector <1 x float> %a, <1 x float> %b, <2 x i32> <i32 0, i32 1>
  store <2 x float> %concat, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2f32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2f32(<4 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) nounwind {
  %concat = shufflevector <2 x float> %a, <2 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x float> %concat, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4f32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4f32(<8 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b) nounwind {
  %concat = shufflevector <4 x float> %a, <4 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x float> %concat, <8 x float> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8f32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8f32(<16 x float> addrspace(1)* %out, <8 x float> %a, <8 x float> %b) nounwind {
  %concat = shufflevector <8 x float> %a, <8 x float> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x float> %concat, <16 x float> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16f32:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16f32(<32 x float> addrspace(1)* %out, <16 x float> %a, <16 x float> %b) nounwind {
  %concat = shufflevector <16 x float> %a, <16 x float> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x float> %concat, <32 x float> addrspace(1)* %out, align 128
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v1i64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1i64(<2 x double> addrspace(1)* %out, <1 x double> %a, <1 x double> %b) nounwind {
  %concat = shufflevector <1 x double> %a, <1 x double> %b, <2 x i32> <i32 0, i32 1>
  store <2 x double> %concat, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2i64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2i64(<4 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b) nounwind {
  %concat = shufflevector <2 x double> %a, <2 x double> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x double> %concat, <4 x double> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4i64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4i64(<8 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b) nounwind {
  %concat = shufflevector <4 x double> %a, <4 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x double> %concat, <8 x double> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8i64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8i64(<16 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b) nounwind {
  %concat = shufflevector <8 x double> %a, <8 x double> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x double> %concat, <16 x double> addrspace(1)* %out, align 128
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16i64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16i64(<32 x double> addrspace(1)* %out, <16 x double> %a, <16 x double> %b) nounwind {
  %concat = shufflevector <16 x double> %a, <16 x double> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x double> %concat, <32 x double> addrspace(1)* %out, align 256
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v1f64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1f64(<2 x double> addrspace(1)* %out, <1 x double> %a, <1 x double> %b) nounwind {
  %concat = shufflevector <1 x double> %a, <1 x double> %b, <2 x i32> <i32 0, i32 1>
  store <2 x double> %concat, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2f64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2f64(<4 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b) nounwind {
  %concat = shufflevector <2 x double> %a, <2 x double> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x double> %concat, <4 x double> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4f64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4f64(<8 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b) nounwind {
  %concat = shufflevector <4 x double> %a, <4 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x double> %concat, <8 x double> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8f64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8f64(<16 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b) nounwind {
  %concat = shufflevector <8 x double> %a, <8 x double> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x double> %concat, <16 x double> addrspace(1)* %out, align 128
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16f64:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16f64(<32 x double> addrspace(1)* %out, <16 x double> %a, <16 x double> %b) nounwind {
  %concat = shufflevector <16 x double> %a, <16 x double> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x double> %concat, <32 x double> addrspace(1)* %out, align 256
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v1i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1i1(<2 x i1> addrspace(1)* %out, <1 x i1> %a, <1 x i1> %b) nounwind {
  %concat = shufflevector <1 x i1> %a, <1 x i1> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i1> %concat, <2 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2i1(<4 x i1> addrspace(1)* %out, <2 x i1> %a, <2 x i1> %b) nounwind {
  %concat = shufflevector <2 x i1> %a, <2 x i1> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i1> %concat, <4 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4i1(<8 x i1> addrspace(1)* %out, <4 x i1> %a, <4 x i1> %b) nounwind {
  %concat = shufflevector <4 x i1> %a, <4 x i1> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i1> %concat, <8 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8i1(<16 x i1> addrspace(1)* %out, <8 x i1> %a, <8 x i1> %b) nounwind {
  %concat = shufflevector <8 x i1> %a, <8 x i1> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i1> %concat, <16 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16i1(<32 x i1> addrspace(1)* %out, <16 x i1> %a, <16 x i1> %b) nounwind {
  %concat = shufflevector <16 x i1> %a, <16 x i1> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i1> %concat, <32 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v32i1:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v32i1(<64 x i1> addrspace(1)* %out, <32 x i1> %a, <32 x i1> %b) nounwind {
  %concat = shufflevector <32 x i1> %a, <32 x i1> %b, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x i1> %concat, <64 x i1> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v1i16:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v1i16(<2 x i16> addrspace(1)* %out, <1 x i16> %a, <1 x i16> %b) nounwind {
  %concat = shufflevector <1 x i16> %a, <1 x i16> %b, <2 x i32> <i32 0, i32 1>
  store <2 x i16> %concat, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v2i16:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v2i16(<4 x i16> addrspace(1)* %out, <2 x i16> %a, <2 x i16> %b) nounwind {
  %concat = shufflevector <2 x i16> %a, <2 x i16> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %concat, <4 x i16> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v4i16:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v4i16(<8 x i16> addrspace(1)* %out, <4 x i16> %a, <4 x i16> %b) nounwind {
  %concat = shufflevector <4 x i16> %a, <4 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i16> %concat, <8 x i16> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v8i16:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v8i16(<16 x i16> addrspace(1)* %out, <8 x i16> %a, <8 x i16> %b) nounwind {
  %concat = shufflevector <8 x i16> %a, <8 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i16> %concat, <16 x i16> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_concat_v16i16:
; SI-NOT: s_mov_b32 s{{[0-9]}}, 0x80f000
; SI-NOT: movrel
define void @test_concat_v16i16(<32 x i16> addrspace(1)* %out, <16 x i16> %a, <16 x i16> %b) nounwind {
  %concat = shufflevector <16 x i16> %a, <16 x i16> %b, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i16> %concat, <32 x i16> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}concat_vector_crash:
; SI: s_endpgm
define void @concat_vector_crash(<8 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in) {
bb:
  %tmp = load <2 x float>, <2 x float> addrspace(1)* %in, align 4
  %tmp1 = shufflevector <2 x float> %tmp, <2 x float> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %tmp2 = shufflevector <8 x float> undef, <8 x float> %tmp1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  store <8 x float> %tmp2, <8 x float> addrspace(1)* %out, align 32
  ret void
}
