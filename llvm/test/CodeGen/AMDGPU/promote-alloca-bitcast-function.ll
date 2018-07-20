; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck %s

; FIXME: Error is misleading because it's not an indirect call.

; CHECK: error: <unknown>:0:0: in function crash_call_constexpr_cast void (): unsupported indirect call to function foo

; Make sure that AMDGPUPromoteAlloca doesn't crash if the called
; function is a constantexpr cast of a function.

declare void @foo(float addrspace(5)*) #0
declare void @foo.varargs(...) #0

; XCHECK: in function crash_call_constexpr_cast{{.*}}: unsupported call to function foo
define amdgpu_kernel void @crash_call_constexpr_cast() #0 {
  %alloca = alloca i32, addrspace(5)
  call void bitcast (void (float addrspace(5)*)* @foo to void (i32 addrspace(5)*)*)(i32 addrspace(5)* %alloca) #0
  ret void
}

; XCHECK: in function crash_call_constexpr_cast{{.*}}: unsupported call to function foo.varargs
define amdgpu_kernel void @crash_call_constexpr_cast_varargs() #0 {
  %alloca = alloca i32, addrspace(5)
  call void bitcast (void (...)* @foo.varargs to void (i32 addrspace(5)*)*)(i32 addrspace(5)* %alloca) #0
  ret void
}

attributes #0 = { nounwind }
