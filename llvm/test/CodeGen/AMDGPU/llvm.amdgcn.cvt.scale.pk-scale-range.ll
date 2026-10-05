; RUN: not llc -mtriple=amdgpu12.50 %s -filetype=null 2>&1 | FileCheck -check-prefix=GFX1250-ERR %s
; RUN: not llc -mtriple=amdgpu12.50s %s -filetype=null 2>&1 | FileCheck -check-prefix=GFX1250S-ERR %s

declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_fp8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_bf8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f16_bf8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_fp8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_bf8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_bf16_bf8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_fp4_scalesel_4 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp4_scalesel_4(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %src, i32 %scale, i32 4)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_fp4_scalesel_4 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp4_scalesel_4(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 4)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_fp8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_bf8_scalesel_8 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f32_bf8_scalesel_8(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %src, i32 %scale, i32 8)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_fp4_scalesel_4 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp4_scalesel_4(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %src, i32 %scale, i32 4)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 32
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_fp6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f16_fp6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_fp6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_bf16_fp6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_bf6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f16_bf6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_bf6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_bf16_bf6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_fp6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f32_fp6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_bf6_scalesel_4 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f32_bf6_scalesel_4(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_fp8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_bf8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f16_bf8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_fp8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_bf8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_bf16_bf8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.fp4 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f16_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f16.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp4_scalesel_8(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %src, i32 %scale, i32 8)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.fp4 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_bf16_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.bf16.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp4_scalesel_8(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 8)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_fp8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.fp8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_bf8_scalesel_15 void (<2 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.bf8 scale_sel maximum supported value is 7

define amdgpu_ps void @test_cvt_scale_pk8_f32_bf8_scalesel_15(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %src, i32 %scale, i32 15)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.fp4 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk8_f32_fp4_scalesel_8 void (i32, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk8.f32.fp4 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp4_scalesel_8(i32 %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %src, i32 %scale, i32 8)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 32
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.fp6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f16_fp6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.fp6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_bf16_fp6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.bf6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f16_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f16.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f16_bf6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.bf6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_bf16_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.bf16.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_bf16_bf6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.fp6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_fp6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.fp6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f32_fp6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

; GFX1250-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.bf6 scale_sel maximum supported value is 7
; GFX1250S-ERR-DAG: error: <unknown>:0:0: in function test_cvt_scale_pk16_f32_bf6_scalesel_8 void (<3 x i32>, i32, ptr addrspace(1)): llvm.amdgcn.cvt.scale.pk16.f32.bf6 scale_sel maximum supported value is 3

define amdgpu_ps void @test_cvt_scale_pk16_f32_bf6_scalesel_8(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %src, i32 %scale, i32 8)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

