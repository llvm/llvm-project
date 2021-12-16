; RUN: not llc -march=amdgcn -mtriple=amdgcn-mesa-mesa3d -tailcallopt < %s 2>&1 | FileCheck --check-prefix=GCN %s
; RUN: not llc -march=amdgcn -mtriple=amdgcn--amdpal -tailcallopt < %s 2>&1 | FileCheck --check-prefix=GCN %s
; RUN: not llc -march=r600 -mtriple=r600-- -mcpu=cypress -tailcallopt < %s 2>&1 | FileCheck -check-prefix=R600 %s

declare i32 @external_function(i32) nounwind

; GCN-NOT: error
; R600: in function test_call_external{{.*}}: unsupported call to function external_function
define amdgpu_kernel void @test_call_external(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = call i32 @external_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define i32 @defined_function(i32 %x) nounwind noinline {
  %y = add i32 %x, 8
  ret i32 %y
}

; GCN-NOT: error
; R600: in function test_call{{.*}}: unsupported call to function defined_function
define amdgpu_kernel void @test_call(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = call i32 @defined_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN: error: <unknown>:0:0: in function test_tail_call i32 (i32 addrspace(1)*, i32 addrspace(1)*): unsupported required tail call to function defined_function
; R600: in function test_tail_call{{.*}}: unsupported call to function defined_function
define i32 @test_tail_call(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = tail call i32 @defined_function(i32 %b)
  ret i32 %c
}

declare void @external.varargs(i32, double, i64, ...)

; GCN: error: <unknown>:0:0: in function test_call_varargs void (): unsupported call to variadic function external.varargs
; R600: in function test_call_varargs{{.*}}: unsupported call to function external.varargs
define void @test_call_varargs() {
  call void (i32, double, i64, ...) @external.varargs(i32 42, double 1.0, i64 12, i8 3, i16 1, i32 4, float 1.0, double 2.0)
  ret void
}

declare i32 @extern_variadic(...)

; GCN: in function test_tail_call_bitcast_extern_variadic{{.*}}: unsupported call to variadic function extern_variadic
; R600: in function test_tail_call_bitcast_extern_variadic{{.*}}: unsupported call to function extern_variadic
define i32 @test_tail_call_bitcast_extern_variadic(<4 x float> %arg0, <4 x float> %arg1, i32 %arg2) {
  %add = fadd <4 x float> %arg0, %arg1
  %call = tail call i32 bitcast (i32 (...)* @extern_variadic to i32 (<4 x float>)*)(<4 x float> %add)
  ret i32 %call
}

; GCN: :0:0: in function test_c_call_from_shader i32 (): unsupported calling convention for call from graphics shader of function defined_function
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

