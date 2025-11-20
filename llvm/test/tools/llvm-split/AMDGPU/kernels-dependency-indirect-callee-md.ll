; RUN: sed 's/_MD_/, !callees !{ptr @CallCandidate0}/g' %s | llvm-split -o %t -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; RUN: sed 's/_MD_//g' %s | llvm-split -o %t-nomd -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t-nomd0 | FileCheck --check-prefix=CHECK-NOMD0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t-nomd1 | FileCheck --check-prefix=CHECK-NOMD1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t-nomd2 | FileCheck --check-prefix=CHECK-NOMD2 --implicit-check-not=define %s

; CHECK0: define internal void @HelperC
; CHECK0: define amdgpu_kernel void @C

; CHECK1: define hidden void @CallCandidate1
; CHECK1: define internal void @HelperB
; CHECK1: define amdgpu_kernel void @B

; CHECK2: define internal void @HelperA
; CHECK2: define hidden void @CallCandidate0
; CHECK2: define amdgpu_kernel void @A

; CHECK-NOMD0: define internal void @HelperC
; CHECK-NOMD0: define amdgpu_kernel void @C

; CHECK-NOMD1: define internal void @HelperB
; CHECK-NOMD1: define amdgpu_kernel void @B

; CHECK-NOMD2: define internal void @HelperA
; CHECK-NOMD2: define hidden void @CallCandidate0
; CHECK-NOMD2: define hidden void @CallCandidate1
; CHECK-NOMD2: define amdgpu_kernel void @A

@addrthief = global [2 x ptr] [ptr @CallCandidate0, ptr @CallCandidate1]

define internal void @HelperA(ptr %call) {
  call void %call() _MD_
  ret void
}

define internal void @CallCandidate0() {
  ret void
}

define internal void @CallCandidate1() {
  ret void
}

define internal void @HelperB() {
  ret void
}

define internal void @HelperC() {
  ret void
}

define amdgpu_kernel void @A(ptr %call) {
  call void @HelperA(ptr %call)
  ret void
}

define amdgpu_kernel void @B() {
  call void @HelperB()
  ret void
}

define amdgpu_kernel void @C() {
  call void @HelperC()
  ret void
}
