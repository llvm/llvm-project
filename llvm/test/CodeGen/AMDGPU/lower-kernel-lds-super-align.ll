; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=true --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=true --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s

; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [32 x i8] }
; CHECK: %llvm.amdgcn.kernel.k2.lds.t = type { i16, [2 x i8], i16 }
; CHECK: %llvm.amdgcn.kernel.k3.lds.t = type { [32 x i64], [32 x i32] }
; CHECK: %llvm.amdgcn.kernel.k4.lds.t = type { [2 x ptr addrspace(3)] }

; SUPER-ALIGN_ON: @lds.unused = addrspace(3) global i32 undef, align 4
; SUPER-ALIGN_OFF: @lds.unused = addrspace(3) global i32 undef, align 2
@lds.unused = addrspace(3) global i32 undef, align 2

@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @lds.unused to ptr)], section "llvm.metadata"

; CHECK-NOT: @lds.1
@lds.1 = internal unnamed_addr addrspace(3) global [32 x i8] undef, align 1

; SUPER-ALIGN_ON: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
; SUPER-ALIGN_OFF: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 1

; CHECK: @llvm.amdgcn.kernel.k2.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k2.lds.t undef, align 4
; SUPER-ALIGN_ON:  @llvm.amdgcn.kernel.k3.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k3.lds.t undef, align 16
; SUPER-ALIGN_OFF: @llvm.amdgcn.kernel.k3.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k3.lds.t undef, align 8

; SUPER-ALIGN_ON:  @llvm.amdgcn.kernel.k4.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k4.lds.t undef, align 16
; SUPER-ALIGN_OFF: @llvm.amdgcn.kernel.k4.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k4.lds.t undef, align 4

; CHECK-LABEL: @k1
; CHECK:  %1 = addrspacecast ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds to ptr
; CHECK:  %ptr = getelementptr inbounds i8, ptr %1, i64 %x
; CHECK:  store i8 1, ptr %ptr, align 1
define amdgpu_kernel void @k1(i64 %x) {
  %ptr = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(3) @lds.1 to ptr), i64 %x
  store i8 1, ptr addrspace(0) %ptr, align 1
  ret void
}

@lds.2 = internal unnamed_addr addrspace(3) global i16 undef, align 4
@lds.3 = internal unnamed_addr addrspace(3) global i16 undef, align 4

; Check that alignment is propagated to uses for scalar variables.

; CHECK-LABEL: @k2
; CHECK: store i16 1, ptr addrspace(3) @llvm.amdgcn.kernel.k2.lds, align 4
; CHECK: store i16 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k2.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k2.lds, i32 0, i32 2), align 4
define amdgpu_kernel void @k2() {
  store i16 1, ptr addrspace(3) @lds.2, align 2
  store i16 2, ptr addrspace(3) @lds.3, align 2
  ret void
}

@lds.4 = internal unnamed_addr addrspace(3) global [32 x i64] undef, align 8
@lds.5 = internal unnamed_addr addrspace(3) global [32 x i32] undef, align 4

; Check that alignment is propagated to uses for arrays.

; CHECK-LABEL: @k3
; CHECK:  store i32 1, ptr addrspace(3) %ptr1, align 8
; CHECK:  store i32 2, ptr addrspace(3) %ptr2, align 4
; SUPER-ALIGN_ON:  store i32 3, ptr addrspace(3) %ptr3, align 16
; SUPER-ALIGN_OFF: store i32 3, ptr addrspace(3) %ptr3, align 8
; CHECK:  store i32 4, ptr addrspace(3) %ptr4, align 4
; CHECK:  store i32 5, ptr addrspace(3) %ptr5, align 4
; CHECK:  %load1 = load i32, ptr addrspace(3) %ptr1, align 8
; CHECK:  %load2 = load i32, ptr addrspace(3) %ptr2, align 4
; SUPER-ALIGN_ON:   %load3 = load i32, ptr addrspace(3) %ptr3, align 16
; SUPER-ALIGN_OFF:  %load3 = load i32, ptr addrspace(3) %ptr3, align 8
; CHECK:  %load4 = load i32, ptr addrspace(3) %ptr4, align 4
; CHECK:  %load5 = load i32, ptr addrspace(3) %ptr5, align 4
; CHECK:  %val1 = atomicrmw volatile add ptr addrspace(3) %ptr1, i32 1 monotonic, align 8
; CHECK:  %val2 = cmpxchg volatile ptr addrspace(3) %ptr1, i32 1, i32 2 monotonic monotonic, align 8
; CHECK:  store i16 11, ptr addrspace(3) %ptr1, align 8
; CHECK:  store i16 12, ptr addrspace(3) %ptr2, align 4
; SUPER-ALIGN_ON:   store i16 13, ptr addrspace(3) %ptr3, align 16
; SUPER-ALIGN_OFF:  store i16 13, ptr addrspace(3) %ptr3, align 8
; CHECK:  store i16 14, ptr addrspace(3) %ptr4, align 4
; CHECK:  %ptr1.ac = addrspacecast ptr addrspace(3) %ptr1 to ptr
; CHECK:  %ptr2.ac = addrspacecast ptr addrspace(3) %ptr2 to ptr
; CHECK:  %ptr3.ac = addrspacecast ptr addrspace(3) %ptr3 to ptr
; CHECK:  %ptr4.ac = addrspacecast ptr addrspace(3) %ptr4 to ptr
; CHECK:  store i32 21, ptr %ptr1.ac, align 8
; CHECK:  store i32 22, ptr %ptr2.ac, align 4
; SUPER-ALIGN_ON:   store i32 23, ptr %ptr3.ac, align 16
; SUPER-ALIGN_OFF:  store i32 23, ptr %ptr3.ac, align 8
; CHECK:  store i32 24, ptr %ptr4.ac, align 4
define amdgpu_kernel void @k3(i64 %x) {
  store i64 0, ptr addrspace(3) @lds.4, align 8

  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) @lds.5, i64 2
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) @lds.5, i64 3
  %ptr3 = getelementptr inbounds i32, ptr addrspace(3) @lds.5, i64 4
  %ptr4 = getelementptr inbounds i32, ptr addrspace(3) @lds.5, i64 5
  %ptr5 = getelementptr inbounds i32, ptr addrspace(3) @lds.5, i64 %x

  store i32 1, ptr addrspace(3) %ptr1, align 4
  store i32 2, ptr addrspace(3) %ptr2, align 4
  store i32 3, ptr addrspace(3) %ptr3, align 4
  store i32 4, ptr addrspace(3) %ptr4, align 4
  store i32 5, ptr addrspace(3) %ptr5, align 4

  %load1 = load i32, ptr addrspace(3) %ptr1, align 4
  %load2 = load i32, ptr addrspace(3) %ptr2, align 4
  %load3 = load i32, ptr addrspace(3) %ptr3, align 4
  %load4 = load i32, ptr addrspace(3) %ptr4, align 4
  %load5 = load i32, ptr addrspace(3) %ptr5, align 4

  %val1 = atomicrmw volatile add ptr addrspace(3) %ptr1, i32 1 monotonic, align 4
  %val2 = cmpxchg volatile ptr addrspace(3) %ptr1, i32 1, i32 2 monotonic monotonic, align 4


  store i16 11, ptr addrspace(3) %ptr1, align 2
  store i16 12, ptr addrspace(3) %ptr2, align 2
  store i16 13, ptr addrspace(3) %ptr3, align 2
  store i16 14, ptr addrspace(3) %ptr4, align 2

  %ptr1.ac = addrspacecast ptr addrspace(3) %ptr1 to ptr
  %ptr2.ac = addrspacecast ptr addrspace(3) %ptr2 to ptr
  %ptr3.ac = addrspacecast ptr addrspace(3) %ptr3 to ptr
  %ptr4.ac = addrspacecast ptr addrspace(3) %ptr4 to ptr

  store i32 21, ptr %ptr1.ac, align 4
  store i32 22, ptr %ptr2.ac, align 4
  store i32 23, ptr %ptr3.ac, align 4
  store i32 24, ptr %ptr4.ac, align 4

  ret void
}

@lds.6 = internal unnamed_addr addrspace(3) global [2 x ptr addrspace(3)] undef, align 4

; Check that aligment is not propagated if use is not a pointer operand.

; CHECK-LABEL: @k4
; SUPER-ALIGN_ON:  store i32 undef, ptr addrspace(3) %gep, align 8
; SUPER-ALIGN_OFF: store i32 undef, ptr addrspace(3) %gep, align 4
; CHECK:           store ptr addrspace(3) %gep, ptr undef, align 4
; SUPER-ALIGN_ON:  %val1 = cmpxchg volatile ptr addrspace(3) %gep, i32 1, i32 2 monotonic monotonic, align 8
; SUPER-ALIGN_OFF: %val1 = cmpxchg volatile ptr addrspace(3) %gep, i32 1, i32 2 monotonic monotonic, align 4
; CHECK:           %val2 = cmpxchg volatile ptr undef, ptr addrspace(3) %gep, ptr addrspace(3) undef monotonic monotonic, align 4
define amdgpu_kernel void @k4() {
  %gep = getelementptr inbounds ptr addrspace(3), ptr addrspace(3) @lds.6, i64 1
  store i32 undef, ptr addrspace(3) %gep, align 4
  store ptr addrspace(3) %gep, ptr undef, align 4
  %val1 = cmpxchg volatile ptr addrspace(3) %gep, i32 1, i32 2 monotonic monotonic, align 4
  %val2 = cmpxchg volatile ptr undef, ptr addrspace(3) %gep, ptr addrspace(3) undef monotonic monotonic, align 4
  ret void
}
