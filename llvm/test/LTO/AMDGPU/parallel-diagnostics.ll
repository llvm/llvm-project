; REQUIRES: amdgpu-registered-target

; Check that diagnostics emitted while code-generating the parallel partitions
; (-j > 1) are reported instead of crashing. The kernel resource usage remarks
; are emitted by the AMDGPU backend during codegen. There is a single kernel so
; that only one partition emits remarks and the output is not interleaved.

; RUN: rm -rf %t && mkdir %t
; RUN: llvm-as -o %t.bc %s

; Without a cache.
; RUN: llvm-lto -exported-symbol=A -j2 -pass-remarks-analysis=kernel-resource-usage -o %t/nocache.o %t.bc 2>&1 | FileCheck %s
; RUN: llvm-nm %t/nocache.o.0 | FileCheck %s --check-prefix=SYMS

; With a cache, on a miss.
; RUN: llvm-lto -exported-symbol=A -j2 -pass-remarks-analysis=kernel-resource-usage -thinlto-cache-dir=%t/cache -o %t/miss.o %t.bc 2>&1 | FileCheck %s
; RUN: llvm-nm %t/miss.o.0 | FileCheck %s --check-prefix=SYMS

; CHECK: remark: <unknown>:0:0: Function Name: A

; SYMS: T A

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu9.00-amd-amdhsa"

define amdgpu_kernel void @A() {
  ret void
}
