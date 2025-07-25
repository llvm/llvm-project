; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; --------------------------------------------------------------------
; Wrong mangled types
; --------------------------------------------------------------------

; CHECK: operand 1 must be 8, 12 or 16 element i32 vector
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i64.v16i32(i32 0, <16 x i64> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <16 x i64> %A
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i64_fp8___v16i32_fp8(<16 x i64> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i64.v16i32(i32 0, <16 x i64> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: operand 3 must be 8, 12 or 16 element i32 vector
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i64(i32 0, <16 x i32> %A, i32 0, <16 x i64> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <16 x i64> %B
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v16i64_fp8(<16 x i32> %A, <16 x i64> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i64(i32 0, <16 x i32> %A, i32 0, <16 x i64> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------
; Impossible vector types
; --------------------------------------------------------------------

; CHECK: operand 1 must be 8, 12 or 16 element i32 vector
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v9i32.v16i32(i32 0, <9 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <9 x i32> %A
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v9i32_fp8___v16i32_fp8(<9 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v9i32.v16i32(i32 0, <9 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: operand 3 must be 8, 12 or 16 element i32 vector
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v9i32(i32 0, <16 x i32> %A, i32 0, <9 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <9 x i32> %B
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v9i32_fp8(<16 x i32> %A, <9 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v9i32(i32 0, <16 x i32> %A, i32 0, <9 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------
; Out of bounds format
; --------------------------------------------------------------------

; CHECK: invalid value for matrix format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 9999, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: i32 9999
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_invalid0___v16i32_fp8(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 9999, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid value for matrix format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 9999, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: i32 9999
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v16i32_invalid1(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 9999, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------
; Incorrect signature for format cases (IR vector too small)
; --------------------------------------------------------------------

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 0, <8 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <8 x i32> %A
; CHECK-NEXT: i32 0
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v8i32_fp8___v16i32_fp8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 0, <8 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v12i32.v16i32(i32 0, <12 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <12 x i32> %A
; CHECK-NEXT: i32 0
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v12i32_fp8___v16i32_fp8(<12 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v12i32.v16i32(i32 0, <12 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 1, <8 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <8 x i32> %A
; CHECK-NEXT: i32 1
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v8i32_bf8___v16i32_fp8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 1, <8 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v12i32.v16i32(i32 1, <12 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <12 x i32> %A
; CHECK-NEXT: i32 1
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v12i32_bf8___v16i32_fp8(<12 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v12i32.v16i32(i32 1, <12 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v8i32(i32 0, <16 x i32> %A, i32 0, <8 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <8 x i32> %B
; CHECK-NEXT: i32 0
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v8i32_fp8(<16 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v8i32(i32 0, <16 x i32> %A, i32 0, <8 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v12i32(i32 0, <16 x i32> %A, i32 0, <12 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <12 x i32> %B
; CHECK-NEXT: i32 0
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v12i32_fp8(<16 x i32> %A, <12 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v12i32(i32 0, <16 x i32> %A, i32 0, <12 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v8i32(i32 0, <16 x i32> %A, i32 1, <8 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <8 x i32> %B
; CHECK-NEXT: i32 1
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v8i32_bf8(<16 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v8i32(i32 0, <16 x i32> %A, i32 1, <8 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v12i32(i32 0, <16 x i32> %A, i32 1, <12 x i32> %B, i16 0, <8 x float> %C)
; CHECK-NEXT: <12 x i32> %B
; CHECK-NEXT: i32 1
define amdgpu_ps void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp8___v12i32_bf8(<16 x i32> %A, <12 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v12i32(i32 0, <16 x i32> %A, i32 1, <12 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}
