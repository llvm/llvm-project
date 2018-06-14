; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-opencl-12-adapter -mcpu=fiji -S < %s | FileCheck %s

; CHECK-NOT: define linkonce_odr <2 x i32> @_Z6vload2mPKi

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(1)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

; CHECK-NOT: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS4Ki

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS3Ki(i64, i32 addrspace(3)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(3)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS5Ki(i64, i32 addrspace(5)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(5)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

; CHECK: declare void @_Z10ndrange_2DPKm(%struct.ndrange_t addrspace(5)*, i64*)

; CHECK: declare void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v

%struct.ndrange_t = type { i32, [3 x i64], [3 x i64], [3 x i64] }
%opencl.clk_event_t = type opaque

define amdgpu_kernel void @test_fn() {
entry:

call <2 x i32> @_Z6vload2mPKi(i64 0, i32* null)
call <2 x i32> @_Z6vload2mPU3AS1Ki(i64 0, i32 addrspace(1)* null)
call <2 x i32> @_Z6vload2mPU3AS4Ki(i64 0, i32 addrspace(4)* null)
call <2 x i32> @_Z6vload2mPU3AS3Ki(i64 0, i32 addrspace(3)* null)
call <2 x i32> @_Z6vload2mPU3AS5Ki(i64 0, i32 addrspace(5)* null)

call void @_Z10ndrange_2DPKm(%struct.ndrange_t addrspace(5)* null, i64* null);

call void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v(%opencl.clk_event_t addrspace(1)* null, i32 0, i8 addrspace(1)* null)

ret void
}

declare <2 x i32> @_Z6vload2mPKi(i64, i32*)
declare <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)
declare <2 x i32> @_Z6vload2mPU3AS4Ki(i64, i32 addrspace(4)*)
declare <2 x i32> @_Z6vload2mPU3AS3Ki(i64, i32 addrspace(3)*)
declare <2 x i32> @_Z6vload2mPU3AS5Ki(i64, i32 addrspace(5)*)

declare void @_Z10ndrange_2DPKm(%struct.ndrange_t addrspace(5)*, i64*)
declare void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v(%opencl.clk_event_t addrspace(1)*, i32, i8 addrspace(1)*)

!opencl.ocl.version = !{!0}
!0 = !{i32 1, i32 2}
