; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Check new struct is added to compiler.used and that the replaced variable is removed

; CHECK: %llvm.amdgcn.module.lds.t = type { float }
; CHECK: @ignored = addrspace(1) global i64 0
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8

; CHECK-NOT: @tolower

@tolower = addrspace(3) global float undef, align 8

; A variable that is unchanged by pass
@ignored = addrspace(1) global i64 0


; @ignored still in list, @tolower removed, llvm.amdgcn.module.lds appended
; Start with one value to replace and one to ignore in the .use list

; @ignored still in list, @tolower removed
; CHECK: @llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(1)* bitcast (i64 addrspace(1)* @ignored to i8 addrspace(1)*) to i8*)], section "llvm.metadata"

@llvm.used = appending global [2 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (float addrspace(3)* @tolower to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64 addrspace(1)* @ignored to i8 addrspace(1)*) to i8*)], section "llvm.metadata"

; @ignored still in list, @tolower removed, llvm.amdgcn.module.lds appended
; CHECK: @llvm.compiler.used = appending global [2 x i8*] [i8* addrspacecast (i8 addrspace(1)* bitcast (i64 addrspace(1)* @ignored to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(3)* bitcast (%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds to i8 addrspace(3)*) to i8*)], section "llvm.metadata"

@llvm.compiler.used = appending global [2 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (float addrspace(3)* @tolower to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64 addrspace(1)* @ignored to i8 addrspace(1)*) to i8*)], section "llvm.metadata"

; CHECK-LABEL: @func()
; CHECK: %dec = atomicrmw fsub float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0), float 1.000000e+00 monotonic, align 4
define void @func() {
  %dec = atomicrmw fsub float addrspace(3)* @tolower, float 1.0 monotonic
  %unused0 = atomicrmw add i64 addrspace(1)* @ignored, i64 1 monotonic
  ret void
}
