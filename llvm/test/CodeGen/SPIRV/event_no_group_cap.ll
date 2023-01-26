; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; __kernel void test_fn( const __global char *src)
; {
;     wait_group_events(0, NULL);
; }

; CHECK-NOT: OpCapability Groups
; CHECK: OpGroupWaitEvents

%opencl.event_t = type opaque

define dso_local spir_kernel void @test_fn(i8 addrspace(1)* noundef %src) {
entry:
  %src.addr = alloca i8 addrspace(1)*, align 8
  store i8 addrspace(1)* %src, i8 addrspace(1)** %src.addr, align 8
  call spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32 noundef 0, %opencl.event_t* addrspace(4)* noundef null)
  ret void
}

declare spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32 noundef, %opencl.event_t* addrspace(4)* noundef)
