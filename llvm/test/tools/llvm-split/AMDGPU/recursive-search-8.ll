; RUN: llvm-split -o %t_s3_ %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-max-depth=8
; RUN: llvm-dis -o - %t_s3_0 | FileCheck --check-prefix=SPLIT3-CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s3_1 | FileCheck --check-prefix=SPLIT3-CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s3_2 | FileCheck --check-prefix=SPLIT3-CHECK2 --implicit-check-not=define %s

; RUN: llvm-split -o %t_s5_ %s -j 5 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-max-depth=8
; RUN: llvm-dis -o - %t_s5_0 | FileCheck --check-prefix=SPLIT5-CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s5_1 | FileCheck --check-prefix=SPLIT5-CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s5_2 | FileCheck --check-prefix=SPLIT5-CHECK2 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s5_3 | FileCheck --check-prefix=SPLIT5-CHECK3 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t_s5_4 | FileCheck --check-prefix=SPLIT5-CHECK4 --implicit-check-not=define %s

; Test the specifics of the search algorithm.
; This test will change depending on new heuristics we add or remove.

; --------------------------------------------

; SPLIT3-CHECK0: define amdgpu_kernel void @A()
; SPLIT3-CHECK0: define internal void @HelperA()
; SPLIT3-CHECK0: define amdgpu_kernel void @B()
; SPLIT3-CHECK0: define internal void @HelperB()

; SPLIT3-CHECK1: define amdgpu_kernel void @C()
; SPLIT3-CHECK1: define internal void @HelperC()

; SPLIT3-CHECK2: define internal void @HelperA()
; SPLIT3-CHECK2: define internal void @HelperB()
; SPLIT3-CHECK2: define internal void @HelperC()
; SPLIT3-CHECK2: define amdgpu_kernel void @AB()
; SPLIT3-CHECK2: define amdgpu_kernel void @BC()
; SPLIT3-CHECK2: define amdgpu_kernel void @ABC()

; --------------------------------------------

; SPLIT5-CHECK0: define amdgpu_kernel void @A()
; SPLIT5-CHECK0: define internal void @HelperA()

; SPLIT5-CHECK1: define amdgpu_kernel void @B()
; SPLIT5-CHECK1: define internal void @HelperB()

; SPLIT5-CHECK2: define internal void @HelperB()
; SPLIT5-CHECK2: define internal void @HelperC()
; SPLIT5-CHECK2: define amdgpu_kernel void @BC

; SPLIT5-CHECK3: define amdgpu_kernel void @C()
; SPLIT5-CHECK3: define internal void @HelperC()

; SPLIT5-CHECK4: define internal void @HelperA()
; SPLIT5-CHECK4: define internal void @HelperB()
; SPLIT5-CHECK4: define internal void @HelperC()
; SPLIT5-CHECK4: define amdgpu_kernel void @AB()
; SPLIT5-CHECK4: define amdgpu_kernel void @ABC()

define amdgpu_kernel void @A() {
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  call void @HelperA()
  ret void
}

define internal void @HelperA() {
  store volatile i32 42, ptr null
  store volatile i32 42, ptr null
  ret void
}

define amdgpu_kernel void @B() {
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  call void @HelperB()
  ret void
}

define internal void @HelperB() {
  store volatile i32 42, ptr null
  store volatile i32 42, ptr null
  store volatile i32 42, ptr null
  ret void
}

define amdgpu_kernel void @C() {
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  store volatile i64 42, ptr null
  call void @HelperC()
  ret void
}

define internal void @HelperC() {
  store volatile i32 42, ptr null
  ret void
}

define amdgpu_kernel void @AB() {
  store volatile i32 42, ptr null
  call void @HelperA()
  call void @HelperB()
  ret void
}

define amdgpu_kernel void @BC() {
  store volatile i32 42, ptr null
  store volatile i32 42, ptr null
  call void @HelperB()
  call void @HelperC()
  ret void
}

define amdgpu_kernel void @ABC() {
  call void @HelperA()
  call void @HelperB()
  call void @HelperC()
  ret void
}
