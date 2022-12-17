; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

; CHECK: %llvm.amdgcn.module.lds.t = type { float, float }
; CHECK: %llvm.amdgcn.kernel.timestwo.lds.t = type { float, float }

@a_func = addrspace(3) global float undef, align 4

@kern = addrspace(3) global float undef, align 4

; @a_func is only used from a non-kernel function so is rewritten
; CHECK-NOT: @a_func
; @b_both is used from a non-kernel function so is rewritten
; CHECK-NOT: @b_both
; sorted both < func, so @b_both at null and @a_func at 4
@b_both = addrspace(3) global float undef, align 4

; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 4
; CHECK: @llvm.amdgcn.kernel.timestwo.lds = internal addrspace(3) global %llvm.amdgcn.kernel.timestwo.lds.t undef, align 4

; CHECK-LABEL: @get_func()
; CHECK:       %0 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0) to i32 addrspace(3)*
; CHECK:       %1 = addrspacecast i32 addrspace(3)* %0 to i32*
; CHECK:       %2 = ptrtoint i32* %1 to i64
; CHECK:       %3 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0) to i32 addrspace(3)*
; CHECK:       %4 = addrspacecast i32 addrspace(3)* %3 to i32*
; CHECK:       %5 = ptrtoint i32* %4 to i64
; CHECK:       %6 = add i64 %2, %5
; CHECK:       %7 = inttoptr i64 %6 to i32*
; CHECK:       %8 = load i32, i32* %7, align 4
; CHECK:       ret i32 %8
define i32 @get_func() local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @a_func to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @a_func to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  ret i32 %0
}

; CHECK-LABEL: @set_func(i32 %x)
; CHECK:      %0 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i32 addrspace(3)*
; CHECK:      %1 = addrspacecast i32 addrspace(3)* %0 to i32*
; CHECK:      %2 = ptrtoint i32* %1 to i64
; CHECK:      %3 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i32 addrspace(3)*
; CHECK:      %4 = addrspacecast i32 addrspace(3)* %3 to i32*
; CHECK:      %5 = ptrtoint i32* %4 to i64
; CHECK:      %6 = add i64 %2, %5
; CHECK:      %7 = inttoptr i64 %6 to i32*
; CHECK:      store i32 %x, i32* %7, align 4
; CHECK:      ret void
define void @set_func(i32 %x) local_unnamed_addr #1 {
entry:
  store i32 %x, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @b_both to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @b_both to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  ret void
}

; CHECK-LABEL: @timestwo() #0
; CHECK-NOT: call void @llvm.donothing()

; CHECK:      %1 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, %llvm.amdgcn.kernel.timestwo.lds.t addrspace(3)* @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 0) to i32 addrspace(3)*
; CHECK:      %2 = addrspacecast i32 addrspace(3)* %1 to i32*
; CHECK:      %3 = ptrtoint i32* %2 to i64
; CHECK:      %4 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, %llvm.amdgcn.kernel.timestwo.lds.t addrspace(3)* @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 1) to i32 addrspace(3)*
; CHECK:      %5 = addrspacecast i32 addrspace(3)* %4 to i32*
; CHECK:      %6 = ptrtoint i32* %5 to i64
; CHECK:      %7 = add i64 %3, %6
; CHECK:      %8 = inttoptr i64 %7 to i32*
; CHECK:      %ld = load i32, i32* %8, align 4
; CHECK:      %mul = mul i32 %ld, 2
; CHECK:      %9 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, %llvm.amdgcn.kernel.timestwo.lds.t addrspace(3)* @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 1) to i32 addrspace(3)*
; CHECK:      %10 = addrspacecast i32 addrspace(3)* %9 to i32*
; CHECK:      %11 = ptrtoint i32* %10 to i64
; CHECK:      %12 = bitcast float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.timestwo.lds.t, %llvm.amdgcn.kernel.timestwo.lds.t addrspace(3)* @llvm.amdgcn.kernel.timestwo.lds, i32 0, i32 0) to i32 addrspace(3)*
; CHECK:      %13 = addrspacecast i32 addrspace(3)* %12 to i32*
; CHECK:      %14 = ptrtoint i32* %13 to i64
; CHECK:      %15 = add i64 %11, %14
; CHECK:      %16 = inttoptr i64 %15 to i32*
; CHECK:      store i32 %mul, i32* %16, align 4
; CHECK:      ret void
define amdgpu_kernel void @timestwo() {
  %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @b_both to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @kern to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  %mul = mul i32 %ld, 2
  store i32 %mul, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @kern to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @b_both to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  ret void
}

; CHECK-LABEL: @through_functions()
define amdgpu_kernel void @through_functions() {
  %ld = call i32 @get_func()
  %mul = mul i32 %ld, 4
  call void @set_func(i32 %mul)
  ret void
}

attributes #0 = { "amdgpu-elide-module-lds" }
; CHECK: attributes #0 = { "amdgpu-elide-module-lds" }
