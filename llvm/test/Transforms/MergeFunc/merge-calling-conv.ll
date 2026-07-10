; RUN: opt -S -passes=mergefunc < %s | FileCheck --implicit-check-not=call %s

; Check that no calls are generated for certain calling conventions

@debug = global i32 0

; CHECK: call void @as_normal

define void @normal(i32 %a) unnamed_addr {
  %b = xor i32 %a, 0
  store i32 %b, ptr @debug
  ret void
}

define void @as_normal(i32 %a) unnamed_addr {
  %b = xor i32 %a, 0
  store i32 %b, ptr @debug
  ret void
}

define amdgpu_kernel void @amdgpu_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 1
  store i32 %b, ptr @debug
  ret void
}

define amdgpu_kernel void @as_amdgpu_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 1
  store i32 %b, ptr @debug
  ret void
}

define ptx_kernel void @ptx_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 2
  store i32 %b, ptr @debug
  ret void
}

define ptx_kernel void @as_ptx_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 2
  store i32 %b, ptr @debug
  ret void
}

define spir_kernel void @spir_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 3
  store i32 %b, ptr @debug
  ret void
}

define spir_kernel void @as_spir_kernel(i32 %a) unnamed_addr {
  %b = xor i32 %a, 3
  store i32 %b, ptr @debug
  ret void
}
