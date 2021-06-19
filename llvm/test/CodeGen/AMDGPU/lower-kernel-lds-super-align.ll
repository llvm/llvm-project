; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s

; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [32 x i8] }

; CHECK-NOT: @lds.1
@lds.1 = internal unnamed_addr addrspace(3) global [32 x i8] undef, align 1

; SUPER-ALIGN_ON: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
; SUPER-ALIGN_OFF: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 1

; CHECK-LABEL: @k1
; CHECK:  %1 = getelementptr inbounds [32 x i8], [32 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 0), i32 0, i32 0
; CHECK:  %2 = addrspacecast i8 addrspace(3)* %1 to i8*
; CHECK:  %ptr = getelementptr inbounds i8, i8* %2, i64 %x
; CHECK:  store i8 1, i8* %ptr, align 1
define amdgpu_kernel void @k1(i64 %x) {
  %ptr = getelementptr inbounds i8, i8* addrspacecast ([32 x i8] addrspace(3)* @lds.1 to i8*), i64 %x
  store i8 1, i8 addrspace(0)* %ptr, align 1
  ret void
}
