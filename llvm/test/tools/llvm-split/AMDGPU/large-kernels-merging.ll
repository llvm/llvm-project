; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-max-depth=0 -amdgpu-module-splitting-large-threshold=1.2 -amdgpu-module-splitting-merge-threshold=0.5
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; RUN: llvm-split -o %t.nolarge %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-large-threshold=0 -amdgpu-module-splitting-max-depth=0
; RUN: llvm-dis -o - %t.nolarge0 | FileCheck --check-prefix=NOLARGEKERNELS-CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t.nolarge1 | FileCheck --check-prefix=NOLARGEKERNELS-CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t.nolarge2 | FileCheck --check-prefix=NOLARGEKERNELS-CHECK2 --implicit-check-not=define %s

; 2 kernels (A/B) are large and share all their dependencies.
; They should go in the same partition, the remaining kernel should
; go somewhere else, and one partition should be empty.
;
; Also check w/o large kernels processing to verify they are indeed handled
; differently.

; P0 is empty
; CHECK0: declare

; CHECK1: define internal void @HelperC()
; CHECK1: define amdgpu_kernel void @C

; CHECK2: define internal void @large2()
; CHECK2: define internal void @large1()
; CHECK2: define internal void @large0()
; CHECK2: define internal void @HelperA()
; CHECK2: define internal void @HelperB()
; CHECK2: define amdgpu_kernel void @A
; CHECK2: define amdgpu_kernel void @B

; NOLARGEKERNELS-CHECK0: define internal void @HelperC()
; NOLARGEKERNELS-CHECK0: define amdgpu_kernel void @C

; NOLARGEKERNELS-CHECK1: define internal void @large2()
; NOLARGEKERNELS-CHECK1: define internal void @large1()
; NOLARGEKERNELS-CHECK1: define internal void @large0()
; NOLARGEKERNELS-CHECK1: define internal void @HelperB()
; NOLARGEKERNELS-CHECK1: define amdgpu_kernel void @B

; NOLARGEKERNELS-CHECK2: define internal void @large2()
; NOLARGEKERNELS-CHECK2: define internal void @large1()
; NOLARGEKERNELS-CHECK2: define internal void @large0()
; NOLARGEKERNELS-CHECK2: define internal void @HelperA()
; NOLARGEKERNELS-CHECK2: define amdgpu_kernel void @A


define internal void @large2() {
  store volatile i32 42, ptr null
  call void @large2()
  ret void
}

define internal void @large1() {
  call void @large1()
  call void @large2()
  ret void
}

define internal void @large0() {
  call void @large0()
  call void @large1()
  call void @large2()
  ret void
}

define internal void @HelperA() {
  call void @large0()
  ret void
}

define internal void @HelperB() {
  call void @large0()
  ret void
}

define amdgpu_kernel void @A() {
  call void @HelperA()
  ret void
}

define amdgpu_kernel void @B() {
  call void @HelperB()
  ret void
}

define internal void @HelperC() {
  ret void
}

define amdgpu_kernel void @C() {
  call void @HelperC()
  ret void
}
