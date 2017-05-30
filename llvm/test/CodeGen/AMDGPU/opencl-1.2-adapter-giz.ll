; RUN: opt -mtriple=amdgcn--amdhsa-amdgiz -amdgpu-opencl-12-adapter -mcpu=fiji -S < %s | FileCheck %s

; CHECK-NOT: define linkonce_odr <2 x i32> @_Z6vload2mPKi

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(1)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

; CHECK-NOT: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS2Ki

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS3Ki(i64, i32 addrspace(3)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(3)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

; CHECK-LABEL: define linkonce_odr <2 x i32> @_Z6vload2mPU3AS5Ki(i64, i32 addrspace(5)*)
; CHECK: %[[r2:.*]] = addrspacecast i32 addrspace(5)* %1 to i32*
; CHECK: %[[r3:.*]] = call <2 x i32> @_Z6vload2mPKi(i64 %0, i32* %[[r2]])
; CHECK: ret <2 x i32> %[[r3]]

define amdgpu_kernel void @test_fn() {
entry:

call <2 x i32> @_Z6vload2mPKi(i64 0, i32* null)
call <2 x i32> @_Z6vload2mPU3AS1Ki(i64 0, i32 addrspace(1)* null)
call <2 x i32> @_Z6vload2mPU3AS2Ki(i64 0, i32 addrspace(2)* null)
call <2 x i32> @_Z6vload2mPU3AS3Ki(i64 0, i32 addrspace(3)* null)
call <2 x i32> @_Z6vload2mPU3AS5Ki(i64 0, i32 addrspace(5)* null)
ret void
}

declare <2 x i32> @_Z6vload2mPKi(i64, i32*)
declare <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)
declare <2 x i32> @_Z6vload2mPU3AS2Ki(i64, i32 addrspace(2)*)
declare <2 x i32> @_Z6vload2mPU3AS3Ki(i64, i32 addrspace(3)*)
declare <2 x i32> @_Z6vload2mPU3AS5Ki(i64, i32 addrspace(5)*)

!opencl.ocl.version = !{!0}
!0 = !{i32 1, i32 2}
