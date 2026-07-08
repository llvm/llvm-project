; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-NOT: {{[Ii]}}ntrinsic has incorrect return type
; CHECK: error:{{.*}}Vector of pointers requires SPV_INTEL_masked_gather_scatter extension

define spir_kernel void @test_vector_gep_v2(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <2 x i64> zeroinitializer
  %elem = extractelement <2 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

define spir_kernel void @test_vector_gep_v4(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <4 x i64> zeroinitializer
  %elem = extractelement <4 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
