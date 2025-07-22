;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=fiji  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=fiji -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define amdgpu_kernel void @xor_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load <2 x i32>, ptr addrspace(1) %in0
  %b = load <2 x i32>, ptr addrspace(1) %in1
  %result = xor <2 x i32> %a, %b
  store <2 x i32> %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @xor_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load <4 x i32>, ptr addrspace(1) %in0
  %b = load <4 x i32>, ptr addrspace(1) %in1
  %result = xor <4 x i32> %a, %b
  store <4 x i32> %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @xor_i1(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load float, ptr addrspace(1) %in0
  %b = load float, ptr addrspace(1) %in1
  %acmp = fcmp oge float %a, 0.000000e+00
  %bcmp = fcmp oge float %b, 1.000000e+00
  %xor = xor i1 %acmp, %bcmp
  %result = select i1 %xor, float %a, float %b
  store float %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_xor_i1(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load volatile i1, ptr addrspace(1) %in0
  %b = load volatile i1, ptr addrspace(1) %in1
  %xor = xor i1 %a, %b
  store i1 %xor, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_xor_i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load i32, ptr addrspace(1) %in0
  %b = load i32, ptr addrspace(1) %in1
  %result = xor i32 %a, %b
  store i32 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_xor_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %result = xor i32 %a, %b
  store i32 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_not_i32(ptr addrspace(1) %out, i32 %a) {
  %result = xor i32 %a, -1
  store i32 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_not_i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load i32, ptr addrspace(1) %in0
  %b = load i32, ptr addrspace(1) %in1
  %result = xor i32 %a, -1
  store i32 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_xor_i64(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load i64, ptr addrspace(1) %in0
  %b = load i64, ptr addrspace(1) %in1
  %result = xor i64 %a, %b
  store i64 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_xor_i64(ptr addrspace(1) %out, i64 %a, i64 %b) {
  %result = xor i64 %a, %b
  store i64 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_not_i64(ptr addrspace(1) %out, i64 %a) {
  %result = xor i64 %a, -1
  store i64 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_not_i64(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
  %a = load i64, ptr addrspace(1) %in0
  %b = load i64, ptr addrspace(1) %in1
  %result = xor i64 %a, -1
  store i64 %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @xor_cf(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %a, i64 %b) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = xor i64 %a, %b
  br label %endif

else:
  %2 = load i64, ptr addrspace(1) %in
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_xor_literal_i64(ptr addrspace(1) %out, [8 x i32], i64 %a) {
  %or = xor i64 %a, 4261135838621753
  store i64 %or, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_xor_literal_multi_use_i64(ptr addrspace(1) %out, [8 x i32], i64 %a, i64 %b) {
  %or = xor i64 %a, 4261135838621753
  store i64 %or, ptr addrspace(1) %out

  %foo = add i64 %b, 4261135838621753
  store volatile i64 %foo, ptr addrspace(1) poison
  ret void
}

define amdgpu_kernel void @scalar_xor_inline_imm_i64(ptr addrspace(1) %out, [8 x i32], i64 %a) {
  %or = xor i64 %a, 63
  store i64 %or, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @scalar_xor_neg_inline_imm_i64(ptr addrspace(1) %out, [8 x i32], i64 %a) {
  %or = xor i64 %a, -8
  store i64 %or, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_xor_i64_neg_inline_imm(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 8
  %or = xor i64 %loada, -8
  store i64 %or, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @vector_xor_literal_i64(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %loada = load i64, ptr addrspace(1) %a, align 8
  %or = xor i64 %loada, 22470723082367
  store i64 %or, ptr addrspace(1) %out
  ret void
}
