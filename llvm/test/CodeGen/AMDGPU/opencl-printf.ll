; RUN: opt -mtriple=r600-- -amdgpu-printf-runtime-binding -mcpu=r600 -S < %s | FileCheck --check-prefix=FUNC --check-prefix=R600 %s
; RUN: opt -mtriple=amdgcn-- -amdgpu-printf-runtime-binding -mcpu=fiji -S < %s | FileCheck --check-prefix=FUNC --check-prefix=GCN %s
; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-printf-runtime-binding -mcpu=fiji -S < %s | FileCheck --check-prefix=FUNC --check-prefix=GCN %s
; RUN: opt -mtriple=amdgcn--amdhsa -passes=amdgpu-printf-runtime-binding -mcpu=fiji -S < %s | FileCheck --check-prefix=FUNC --check-prefix=GCN %s

; FUNC-LABEL: @test_kernel(
; R600-LABEL: entry
; R600-NOT: call i8 addrspace(1)* @__printf_alloc
; R600: call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(2)* @.str, i32 0, i32 0), i8 addrspace(5)* %arraydecay, i32 %n)
; GCN-LABEL: entry
; GCN: call i8 addrspace(1)* @__printf_alloc
; GCN-LABEL: entry.split
; GCN: icmp ne i8 addrspace(1)* %printf_alloc_fn, null
; GCN: %PrintBuffID = getelementptr i8, i8 addrspace(1)* %printf_alloc_fn, i32 0
; GCN: %PrintBuffIdCast = bitcast i8 addrspace(1)* %PrintBuffID to i32 addrspace(1)*
; GCN: store i32 1, i32 addrspace(1)* %PrintBuffIdCast
; GCN: %PrintBuffGep = getelementptr i8, i8 addrspace(1)* %printf_alloc_fn, i32 4
; GCN: %PrintArgPtr = ptrtoint i8 addrspace(5)* %arraydecay to i64
; GCN: %PrintBuffPtrCast = bitcast i8 addrspace(1)* %PrintBuffGep to i64 addrspace(1)*
; GCN: store i64 %PrintArgPtr, i64 addrspace(1)* %PrintBuffPtrCast
; GCN: %PrintBuffNextPtr = getelementptr i8, i8 addrspace(1)* %PrintBuffGep, i32 8
; GCN: %PrintBuffPtrCast1 = bitcast i8 addrspace(1)* %PrintBuffNextPtr to i32 addrspace(1)*
; GCN: store i32 %n, i32 addrspace(1)* %PrintBuffPtrCast1

@.str = private unnamed_addr addrspace(2) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %arraydecay = getelementptr inbounds [9 x i8], [9 x i8] addrspace(5)* %str, i32 0, i32 0
  %call1 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(2)* @.str, i32 0, i32 0), i8 addrspace(5)* %arraydecay, i32 %n)
  ret void
}

declare i32 @printf(i8 addrspace(2)*, ...)
