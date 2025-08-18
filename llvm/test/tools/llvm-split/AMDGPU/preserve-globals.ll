; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-max-depth=0 -amdgpu-module-splitting-large-threshold=1.2 -amdgpu-module-splitting-merge-threshold=0.5
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s

; Only 2 out of 3 partitions are created, check the external global is preserved in the first partition.

; CHECK0: @foobar = linkonce_odr global i64 52
; CHECK0: define amdgpu_kernel void @B

; CHECK1-NOT: @foobar = linkonce_odr global i64 52
; CHECK1: define amdgpu_kernel void @A

@foobar = linkonce_odr global i64 52

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B() {
  ret void
}
