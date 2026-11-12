; REQUIRES: amdgpu-registered-target

; Check that the parallel codegen partitions are delivered to the caller both
; on a cache miss and on a cache hit.

; RUN: rm -rf %t && mkdir %t
; RUN: llvm-as -o %t.bc %s

; Cache miss.
; RUN: llvm-lto -exported-symbol=A -exported-symbol=B -j2 -amdgpu-module-splitting-large-threshold=0 -thinlto-cache-dir=%t/cache -o %t/miss.o %t.bc
; RUN: llvm-nm %t/miss.o.0 %t/miss.o.1 | FileCheck %s
; RUN: ls %t/cache/llvmcache-* | count 2

; Cache hit: no new cache entries, same outputs.
; RUN: llvm-lto -exported-symbol=A -exported-symbol=B -j2 -amdgpu-module-splitting-large-threshold=0 -thinlto-cache-dir=%t/cache -o %t/hit.o %t.bc
; RUN: llvm-nm %t/hit.o.0 %t/hit.o.1 | FileCheck %s
; RUN: ls %t/cache/llvmcache-* | count 2

; CHECK-DAG: T A
; CHECK-DAG: T B

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu9.00-amd-amdhsa"

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B() {
  ret void
}
