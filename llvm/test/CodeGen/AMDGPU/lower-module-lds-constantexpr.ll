; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

; CHECK: %llvm.amdgcn.module.lds.t = type { float, float }
; CHECK: %llvm.amdgcn.kernel.timestwo.lds.t = type { float, float }

@a_func = addrspace(3) global float poison, align 4

@kern = addrspace(3) global float poison, align 4

; @a_func is only used from a non-kernel function so is rewritten
; CHECK-NOT: @a_func
; @b_both is used from a non-kernel function so is rewritten
; CHECK-NOT: @b_both
; sorted both < func, so @b_both at null and @a_func at 4
@b_both = addrspace(3) global float poison, align 4

; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t poison, align 4
; CHECK: @llvm.amdgcn.kernel.timestwo.lds = internal addrspace(3) global %llvm.amdgcn.kernel.timestwo.lds.t poison, align 4

; CHECK-LABEL: @get_func()
; CHECK:       %0 = addrspacecast ptr addrspace(3) @llvm.amdgcn.module.lds to ptr
; CHECK:       %1 = ptrtoint ptr %0 to i64
; CHECK:       %2 = addrspacecast ptr addrspace(3) @llvm.amdgcn.module.lds to ptr
; CHECK:       %3 = ptrtoint ptr %2 to i64
; CHECK:       %4 = add i64 %1, %3
; CHECK:       %5 = inttoptr i64 %4 to ptr
; CHECK:       %6 = load i32, ptr %5, align 4
; CHECK:       ret i32 %6
define i32 @get_func() local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @a_func to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @a_func to ptr) to i64)) to ptr), align 4
  ret i32 %0
}

; CHECK-LABEL: @set_func(i32 %x)
; CHECK:      %0 = addrspacecast ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 1) to ptr
; CHECK:      %1 = ptrtoint ptr %0 to i64
; CHECK:      %2 = addrspacecast ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 1) to ptr
; CHECK:      %3 = ptrtoint ptr %2 to i64
; CHECK:      %4 = add i64 %1, %3
; CHECK:      %5 = inttoptr i64 %4 to ptr
; CHECK:      store i32 %x, ptr %5, align 4
; CHECK:      ret void
define void @set_func(i32 %x) {
entry:
  store i32 %x, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @b_both to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @b_both to ptr) to i64)) to ptr), align 4
  ret void
}

; CHECK-LABEL: @timestwo() #0
; CHECK-NOT: call void @llvm.donothing()

; CHECK:      %1 = addrspacecast ptr addrspace(3) @llvm.amdgcn.kernel.timestwo.lds to ptr
; CHECK:      %2 = ptrtoint ptr %1 to i64
; CHECK:      %3 = addrspacecast ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 1) to ptr
; CHECK:      %4 = ptrtoint ptr %3 to i64
; CHECK:      %5 = add i64 %2, %4
; CHECK:      %6 = inttoptr i64 %5 to ptr
; CHECK:      %ld = load i32, ptr %6, align 4
; CHECK:      %mul = mul i32 %ld, 2
; CHECK:      %7 = addrspacecast ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 1) to ptr
; CHECK:      %8 = ptrtoint ptr %7 to i64
; CHECK:      %9 = addrspacecast ptr addrspace(3) @llvm.amdgcn.kernel.timestwo.lds to ptr
; CHECK:      %10 = ptrtoint ptr %9 to i64
; CHECK:      %11 = add i64 %8, %10
; CHECK:      %12 = inttoptr i64 %11 to ptr
; CHECK:      store i32 %mul, ptr %12, align 4
; CHECK:      ret void
define amdgpu_kernel void @timestwo() {
  %ld = load i32, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @b_both to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @kern to ptr) to i64)) to ptr), align 4
  %mul = mul i32 %ld, 2
  store i32 %mul, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @kern to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @b_both to ptr) to i64)) to ptr), align 4
  ret void
}

; CHECK-LABEL: @through_functions() #0
define amdgpu_kernel void @through_functions() {
  %ld = call i32 @get_func()
  %mul = mul i32 %ld, 4
  call void @set_func(i32 %mul)
  ret void
}

; CHECK: attributes #0 = { "amdgpu-lds-size"="8" }
