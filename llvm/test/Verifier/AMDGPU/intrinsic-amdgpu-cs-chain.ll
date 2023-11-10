; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.amdgcn.cs.chain(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) noreturn
declare i32 @llvm.amdgcn.set.inactive.chain.arg(i32, i32) convergent willreturn nofree nocallback readnone

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
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %unused = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 0, i32 1)

  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_kernel void @bad_caller_amdgpu_kernel(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %unused = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 0, i32 1)

  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_gfx void @bad_caller_amdgpu_gfx(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %unused = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 0, i32 1)

  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_vs void @bad_caller_amdgpu_vs(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %unused = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 0, i32 1)

  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs, amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.cs.chain
  call void(ptr, i32, <4 x i32>, { ptr, <3 x i32> }, i32, ...) @llvm.amdgcn.cs.chain(ptr %fn, i32 %exec, <4 x i32> %sgpr, { ptr, <3 x i32> } %vgpr, i32 0)
  unreachable
}

define amdgpu_cs void @bad_caller_amdgpu_cs(ptr %fn, i32 %exec, <4 x i32> inreg %sgpr, { ptr, <3 x i32> } %vgpr) {
  ; CHECK: Intrinsic can only be used from functions with the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %unused = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 0, i32 1)

  ; Unlike llvm.amdgcn.set.inactive.chain.arg, llvm.amdgcn.cs.chain may be called from amdgpu_cs functions.

  ret void
}

define amdgpu_cs_chain void @set_inactive_chain_arg_sgpr(ptr addrspace(1) %out, i32 %active, i32 inreg %inactive) {
  ; CHECK: Value for inactive lanes must be a VGPR function argument
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %tmp = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 %active, i32 %inactive) #0
  store i32 %tmp, ptr addrspace(1) %out
  ret void
}

define amdgpu_cs_chain void @set_inactive_chain_arg_const(ptr addrspace(1) %out, i32 %active) {
  ; CHECK: Value for inactive lanes must be a function argument
  ; CHECK-NEXT: llvm.amdgcn.set.inactive.chain.arg
  %tmp = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 %active, i32 29) #0
  store i32 %tmp, ptr addrspace(1) %out
  ret void
}

define amdgpu_cs_chain void @set_inactive_chain_arg_computed(ptr addrspace(1) %out, i32 %active) {
  ; CHECK: Value for inactive lanes must be a function argument
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %inactive = add i32 %active, 127
  %tmp = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 %active, i32 %inactive) #0
  store i32 %tmp, ptr addrspace(1) %out
  ret void
}

define amdgpu_cs_chain void @set_inactive_chain_arg_inreg(ptr addrspace(1) %out, i32 %active, i32 %inactive) {
  ; CHECK: Value for inactive lanes must not have the `inreg` attribute
  ; CHECK-NEXT: @llvm.amdgcn.set.inactive.chain.arg
  %tmp = call i32 @llvm.amdgcn.set.inactive.chain.arg(i32 %active, i32 inreg %inactive) #0
  store i32 %tmp, ptr addrspace(1) %out
  ret void
}
