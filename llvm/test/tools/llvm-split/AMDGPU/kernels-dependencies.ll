; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; 3 kernels with each their own dependencies should go into 3
; distinct partitions. Aliases should follow their aliasee's partition
; and not be cleaned up even if unused in that partition.

; CHECK0-NOT: @HelperA_alias = {{.*}}alias
; CHECK0-NOT: define
; CHECK0: define amdgpu_kernel void @C
; CHECK0: define internal void @HelperC
; CHECK0-NOT: @HelperA_alias = {{.*}}alias
; CHECK0-NOT: define

; CHECK1-NOT: @HelperA_alias = {{.*}}alias
; CHECK1-NOT: define
; CHECK1: define amdgpu_kernel void @B
; CHECK1: define internal void @HelperB
; CHECK1-NOT: @HelperA_alias = {{.*}}alias
; CHECK1-NOT: define

; CHECK2: @HelperA_alias = hidden alias void (), ptr @HelperA
; CHECK2-NOT: define
; CHECK2: define amdgpu_kernel void @A
; CHECK2: define hidden void @HelperA
; CHECK2-NOT: define

@HelperA_alias = internal alias void (), ptr @HelperA

define amdgpu_kernel void @A() {
  call void @HelperA()
  ret void
}

define internal void @HelperA() {
  ret void
}

define amdgpu_kernel void @B() {
  call void @HelperB()
  ret void
}

define internal void @HelperB() {
  ret void
}

define amdgpu_kernel void @C() {
  call void @HelperC()
  ret void
}

define internal void @HelperC() {
  ret void
}
