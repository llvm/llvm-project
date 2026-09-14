; RUN: opt -S -mtriple=amdgpu-- -amdgpu-lower-module-lds %s -o %t.ll
; RUN: opt -S -mtriple=amdgpu-- -amdgpu-lower-module-lds %t.ll -o %t.second.ll
; RUN: diff -ub %t.ll %t.second.ll -I ".*ModuleID.*"

; Check AMDGPULowerModuleLDS can run more than once on the same module, and that
; the second run is a no-op, as full LTO reruns it on each codegen partition.

; Kernel-only static LDS. @lds2 pushes @lds to a non-zero offset, so the first
; run leaves a constexpr GEP, which the second run used to re-expand.
@lds = internal unnamed_addr addrspace(3) global i32 poison, align 4
@lds2 = internal unnamed_addr addrspace(3) global i64 poison, align 8

; Called from a function shared by two kernels: module struct plus lookup table.
@lds.function = internal addrspace(3) global [8 x i16] poison, align 2

; Dynamic LDS from a function lowers to per-kernel shadows plus an offset table.
@dynlds.function = external hidden addrspace(3) global [0 x float], align 8

; Dynamic LDS used only from a kernel is left in place.
@dynlds = external addrspace(3) global [0 x i32], align 4

; Never lowered, so these survive the first run without an absolute address.
@const.lds = internal addrspace(3) constant [4 x i32] poison, align 4
@initialized.lds = internal addrspace(3) global i16 0, align 2

; Escapes into a global initializer, so lowering cannot place it either.
@escaped = internal addrspace(3) global i32 poison, align 4
@escape.ptr = addrspace(1) global ptr addrspace(3) @escaped, align 4

define void @helper() {
  store i16 1, ptr addrspace(3) getelementptr inbounds ([8 x i16], ptr addrspace(3) @lds.function, i32 0, i32 3), align 2
  store float 2.0, ptr addrspace(3) @dynlds.function, align 8
  ret void
}

define amdgpu_kernel void @test() {
entry:
  call void @helper()
  store i32 0, ptr addrspace(3) @dynlds
  store i32 1, ptr addrspace(3) @lds
  store i64 2, ptr addrspace(3) @lds2
  store i32 3, ptr addrspace(3) @escaped
  %c = load i32, ptr addrspace(3) getelementptr inbounds ([4 x i32], ptr addrspace(3) @const.lds, i32 0, i32 2), align 4
  store i16 9, ptr addrspace(3) @initialized.lds, align 2
  ret void
}

define amdgpu_kernel void @test2() {
entry:
  call void @helper()
  ret void
}
