; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.amdgcn.cs.chain(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) noreturn

define amdgpu_cs_chain void @bad_flags(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr, i32 %flags) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %flags
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 %flags)
  unreachable
}

define amdgpu_cs_chain void @bad_vgpr_args(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } inreg %vgpr) {
  ; CHECK: VGPR arguments must not have the `inreg` attribute
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } inreg %vgpr, i32 0)
  unreachable
}

define amdgpu_cs_chain void @bad_sgpr_args(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: SGPR arguments must have the `inreg` attribute
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_cs_chain void @bad_exec(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr, i32 %flags) {
  ; CHECK: Intrinsic called with incompatible signature
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, <4 x i32>, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, <4 x i32> %sgpr, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 %flags)
  unreachable
}

define void @bad_caller_default_cc(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_kernel void @bad_caller_amdgpu_kernel(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_gfx void @bad_caller_amdgpu_gfx(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_vs void @bad_caller_amdgpu_vs(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}
