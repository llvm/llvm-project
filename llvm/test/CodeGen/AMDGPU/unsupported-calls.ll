; RUN: not llc -mtriple=amdgcn-mesa-mesa3d -tailcallopt < %s 2>&1 | FileCheck --check-prefix=GCN %s
; RUN: not llc -mtriple=amdgcn--amdpal -tailcallopt < %s 2>&1 | FileCheck --check-prefix=GCN %s
; RUN: not llc -mtriple=r600 -mtriple=r600-- -mcpu=cypress -tailcallopt < %s 2>&1 | FileCheck -check-prefix=R600 %s

declare i32 @external_function(i32) nounwind

; GCN-NOT: error
; R600: in function test_call_external{{.*}}: unsupported call to function external_function
define amdgpu_kernel void @test_call_external(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %a = load i32, ptr addrspace(1) %in
  %b = load i32, ptr addrspace(1) %b_ptr
  %c = call i32 @external_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, ptr addrspace(1) %out
  ret void
}

define i32 @defined_function(i32 %x) nounwind noinline {
  %y = add i32 %x, 8
  ret i32 %y
}

; GCN-NOT: error
; R600: in function test_call{{.*}}: unsupported call to function defined_function
define amdgpu_kernel void @test_call(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %a = load i32, ptr addrspace(1) %in
  %b = load i32, ptr addrspace(1) %b_ptr
  %c = call i32 @defined_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; GCN: error: <unknown>:0:0: in function test_tail_call i32 (ptr addrspace(1), ptr addrspace(1)): unsupported required tail call to function defined_function
; R600: in function test_tail_call{{.*}}: unsupported call to function defined_function
define i32 @test_tail_call(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %a = load i32, ptr addrspace(1) %in
  %b = load i32, ptr addrspace(1) %b_ptr
  %c = tail call i32 @defined_function(i32 %b)
  ret i32 %c
}

; R600: in function test_c_call{{.*}}: unsupported call to function defined_function
define amdgpu_ps i32 @test_c_call_from_shader() {
  %call = call i32 @defined_function(i32 0)
  ret i32 %call
}

; GCN-NOT: in function test_gfx_call{{.*}}unsupported
; R600: in function test_gfx_call{{.*}}: unsupported call to function defined_function
define amdgpu_ps i32 @test_gfx_call_from_shader() {
  %call = call amdgpu_gfx i32 @defined_function(i32 0)
  ret i32 %call
}

