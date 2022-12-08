; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=true --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=true --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_ON %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-super-align-lds-globals=false --amdgpu-lower-module-lds-strategy=module < %s | FileCheck --check-prefixes=CHECK,SUPER-ALIGN_OFF %s

; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [32 x i8] }
; CHECK: %llvm.amdgcn.kernel.k2.lds.t = type { i16, [2 x i8], i16 }
; CHECK: %llvm.amdgcn.kernel.k3.lds.t = type { [32 x i64], [32 x i32] }
; CHECK: %llvm.amdgcn.kernel.k4.lds.t = type { [2 x i32 addrspace(3)*] }

; SUPER-ALIGN_ON: @lds.unused = addrspace(3) global i32 undef, align 4
; SUPER-ALIGN_OFF: @lds.unused = addrspace(3) global i32 undef, align 2
@lds.unused = addrspace(3) global i32 undef, align 2

@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i32 addrspace(3)* @lds.unused to i8*)], section "llvm.metadata"

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
; CHECK:  %1 = getelementptr inbounds [32 x i8], [32 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 0), i32 0, i32 0
; CHECK:  %2 = addrspacecast i8 addrspace(3)* %1 to i8*
; CHECK:  %ptr = getelementptr inbounds i8, i8* %2, i64 %x
; CHECK:  store i8 1, i8* %ptr, align 1
define amdgpu_kernel void @k1(i64 %x) {
  %ptr = getelementptr inbounds i8, i8* addrspacecast ([32 x i8] addrspace(3)* @lds.1 to i8*), i64 %x
  store i8 1, i8 addrspace(0)* %ptr, align 1
  ret void
}

@lds.2 = internal unnamed_addr addrspace(3) global i16 undef, align 4
@lds.3 = internal unnamed_addr addrspace(3) global i16 undef, align 4

; Check that alignment is propagated to uses for scalar variables.

; CHECK-LABEL: @k2
; CHECK: store i16 1, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k2.lds.t, %llvm.amdgcn.kernel.k2.lds.t addrspace(3)* @llvm.amdgcn.kernel.k2.lds, i32 0, i32 0), align 4
; CHECK: store i16 2, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k2.lds.t, %llvm.amdgcn.kernel.k2.lds.t addrspace(3)* @llvm.amdgcn.kernel.k2.lds, i32 0, i32 2), align 4
define amdgpu_kernel void @k2() {
  store i16 1, i16 addrspace(3)* @lds.2, align 2
  store i16 2, i16 addrspace(3)* @lds.3, align 2
  ret void
}

@lds.4 = internal unnamed_addr addrspace(3) global [32 x i64] undef, align 8
@lds.5 = internal unnamed_addr addrspace(3) global [32 x i32] undef, align 4

; Check that alignment is propagated to uses for arrays.

; CHECK-LABEL: @k3
; CHECK:  store i32 1, i32 addrspace(3)* %ptr1, align 8
; CHECK:  store i32 2, i32 addrspace(3)* %ptr2, align 4
; SUPER-ALIGN_ON:  store i32 3, i32 addrspace(3)* %ptr3, align 16
; SUPER-ALIGN_OFF: store i32 3, i32 addrspace(3)* %ptr3, align 8
; CHECK:  store i32 4, i32 addrspace(3)* %ptr4, align 4
; CHECK:  store i32 5, i32 addrspace(3)* %ptr5, align 4
; CHECK:  %load1 = load i32, i32 addrspace(3)* %ptr1, align 8
; CHECK:  %load2 = load i32, i32 addrspace(3)* %ptr2, align 4
; SUPER-ALIGN_ON:   %load3 = load i32, i32 addrspace(3)* %ptr3, align 16
; SUPER-ALIGN_OFF:  %load3 = load i32, i32 addrspace(3)* %ptr3, align 8
; CHECK:  %load4 = load i32, i32 addrspace(3)* %ptr4, align 4
; CHECK:  %load5 = load i32, i32 addrspace(3)* %ptr5, align 4
; CHECK:  %val1 = atomicrmw volatile add i32 addrspace(3)* %ptr1, i32 1 monotonic, align 8
; CHECK:  %val2 = cmpxchg volatile i32 addrspace(3)* %ptr1, i32 1, i32 2 monotonic monotonic, align 8
; CHECK:  %ptr1.bc = bitcast i32 addrspace(3)* %ptr1 to i16 addrspace(3)*
; CHECK:  %ptr2.bc = bitcast i32 addrspace(3)* %ptr2 to i16 addrspace(3)*
; CHECK:  %ptr3.bc = bitcast i32 addrspace(3)* %ptr3 to i16 addrspace(3)*
; CHECK:  %ptr4.bc = bitcast i32 addrspace(3)* %ptr4 to i16 addrspace(3)*
; CHECK:  store i16 11, i16 addrspace(3)* %ptr1.bc, align 8
; CHECK:  store i16 12, i16 addrspace(3)* %ptr2.bc, align 4
; SUPER-ALIGN_ON:   store i16 13, i16 addrspace(3)* %ptr3.bc, align 16
; SUPER-ALIGN_OFF:  store i16 13, i16 addrspace(3)* %ptr3.bc, align 8
; CHECK:  store i16 14, i16 addrspace(3)* %ptr4.bc, align 4
; CHECK:  %ptr1.ac = addrspacecast i32 addrspace(3)* %ptr1 to i32*
; CHECK:  %ptr2.ac = addrspacecast i32 addrspace(3)* %ptr2 to i32*
; CHECK:  %ptr3.ac = addrspacecast i32 addrspace(3)* %ptr3 to i32*
; CHECK:  %ptr4.ac = addrspacecast i32 addrspace(3)* %ptr4 to i32*
; CHECK:  store i32 21, i32* %ptr1.ac, align 8
; CHECK:  store i32 22, i32* %ptr2.ac, align 4
; SUPER-ALIGN_ON:   store i32 23, i32* %ptr3.ac, align 16
; SUPER-ALIGN_OFF:  store i32 23, i32* %ptr3.ac, align 8
; CHECK:  store i32 24, i32* %ptr4.ac, align 4
define amdgpu_kernel void @k3(i64 %x) {
  %ptr0 = getelementptr inbounds i64, i64 addrspace(3)* bitcast ([32 x i64] addrspace(3)* @lds.4 to i64 addrspace(3)*), i64 0
  store i64 0, i64 addrspace(3)* %ptr0, align 8

  %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* bitcast ([32 x i32] addrspace(3)* @lds.5 to i32 addrspace(3)*), i64 2
  %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* bitcast ([32 x i32] addrspace(3)* @lds.5 to i32 addrspace(3)*), i64 3
  %ptr3 = getelementptr inbounds i32, i32 addrspace(3)* bitcast ([32 x i32] addrspace(3)* @lds.5 to i32 addrspace(3)*), i64 4
  %ptr4 = getelementptr inbounds i32, i32 addrspace(3)* bitcast ([32 x i32] addrspace(3)* @lds.5 to i32 addrspace(3)*), i64 5
  %ptr5 = getelementptr inbounds i32, i32 addrspace(3)* bitcast ([32 x i32] addrspace(3)* @lds.5 to i32 addrspace(3)*), i64 %x

  store i32 1, i32 addrspace(3)* %ptr1, align 4
  store i32 2, i32 addrspace(3)* %ptr2, align 4
  store i32 3, i32 addrspace(3)* %ptr3, align 4
  store i32 4, i32 addrspace(3)* %ptr4, align 4
  store i32 5, i32 addrspace(3)* %ptr5, align 4

  %load1 = load i32, i32 addrspace(3)* %ptr1, align 4
  %load2 = load i32, i32 addrspace(3)* %ptr2, align 4
  %load3 = load i32, i32 addrspace(3)* %ptr3, align 4
  %load4 = load i32, i32 addrspace(3)* %ptr4, align 4
  %load5 = load i32, i32 addrspace(3)* %ptr5, align 4

  %val1 = atomicrmw volatile add i32 addrspace(3)* %ptr1, i32 1 monotonic, align 4
  %val2 = cmpxchg volatile i32 addrspace(3)* %ptr1, i32 1, i32 2 monotonic monotonic, align 4

  %ptr1.bc = bitcast i32 addrspace(3)* %ptr1 to i16 addrspace(3)*
  %ptr2.bc = bitcast i32 addrspace(3)* %ptr2 to i16 addrspace(3)*
  %ptr3.bc = bitcast i32 addrspace(3)* %ptr3 to i16 addrspace(3)*
  %ptr4.bc = bitcast i32 addrspace(3)* %ptr4 to i16 addrspace(3)*

  store i16 11, i16 addrspace(3)* %ptr1.bc, align 2
  store i16 12, i16 addrspace(3)* %ptr2.bc, align 2
  store i16 13, i16 addrspace(3)* %ptr3.bc, align 2
  store i16 14, i16 addrspace(3)* %ptr4.bc, align 2

  %ptr1.ac = addrspacecast i32 addrspace(3)* %ptr1 to i32*
  %ptr2.ac = addrspacecast i32 addrspace(3)* %ptr2 to i32*
  %ptr3.ac = addrspacecast i32 addrspace(3)* %ptr3 to i32*
  %ptr4.ac = addrspacecast i32 addrspace(3)* %ptr4 to i32*

  store i32 21, i32* %ptr1.ac, align 4
  store i32 22, i32* %ptr2.ac, align 4
  store i32 23, i32* %ptr3.ac, align 4
  store i32 24, i32* %ptr4.ac, align 4

  ret void
}

@lds.6 = internal unnamed_addr addrspace(3) global [2 x i32 addrspace(3)*] undef, align 4

; Check that aligment is not propagated if use is not a pointer operand.

; CHECK-LABEL: @k4
; SUPER-ALIGN_ON:  store i32 undef, i32 addrspace(3)* %ptr, align 8
; SUPER-ALIGN_OFF: store i32 undef, i32 addrspace(3)* %ptr, align 4
; CHECK:           store i32 addrspace(3)* %ptr, i32 addrspace(3)** undef, align 4
; SUPER-ALIGN_ON:  %val1 = cmpxchg volatile i32 addrspace(3)* %ptr, i32 1, i32 2 monotonic monotonic, align 8
; SUPER-ALIGN_OFF: %val1 = cmpxchg volatile i32 addrspace(3)* %ptr, i32 1, i32 2 monotonic monotonic, align 4
; CHECK:           %val2 = cmpxchg volatile i32 addrspace(3)** undef, i32 addrspace(3)* %ptr, i32 addrspace(3)* undef monotonic monotonic, align 4
define amdgpu_kernel void @k4() {
  %gep = getelementptr inbounds i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* bitcast ([2 x i32 addrspace(3)*] addrspace(3)* @lds.6 to i32 addrspace(3)* addrspace(3)*), i64 1
  %ptr = bitcast i32 addrspace(3)* addrspace(3)* %gep to i32 addrspace(3)*
  store i32 undef, i32 addrspace(3)* %ptr, align 4
  store i32 addrspace(3)* %ptr, i32 addrspace(3)** undef, align 4
  %val1 = cmpxchg volatile i32 addrspace(3)* %ptr, i32 1, i32 2 monotonic monotonic, align 4
  %val2 = cmpxchg volatile i32 addrspace(3)** undef, i32 addrspace(3)* %ptr, i32 addrspace(3)* undef monotonic monotonic, align 4
  ret void
}
