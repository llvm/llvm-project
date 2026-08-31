; Test that occupancy attributes are merged conservatively when a device
; function is reachable from multiple kernels with different constraints.
;
; kernel1: flat_wg=[64,256], waves=[2,8], max_wg=[16,16,1]
; kernel2: flat_wg=[32,128], waves=[4,6], max_wg=[8,8,8]
;
; Expected merge for device_func:
;   FlatWGSizeMin = min(64,32) = 32
;   FlatWGSizeMax = max(256,128) = 256
;   WavesPerEUMin = max(2,4) = 4
;   WavesPerEUMax = max(8,6) = 8
;   MaxNumWGX     = max(16,8) = 16
;   MaxNumWGY     = max(16,8) = 16
;   MaxNumWGZ     = max(1,8)  = 8
;
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/kernels.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/device.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o \
; RUN:   -r %t1.bc,kernel1,px \
; RUN:   -r %t1.bc,kernel2,px \
; RUN:   -r %t1.bc,device_func,l \
; RUN:   -r %t2.bc,device_func,px \
; RUN:   -save-temps
; RUN: llvm-dis %t.o.2.5.precodegen.bc -o - | FileCheck %s

; CHECK: define void @device_func({{.*}}) {{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="32,256"
; CHECK-SAME: "amdgpu-max-num-workgroups"="16,16,8"
; CHECK-SAME: "amdgpu-waves-per-eu"="4,8"

;--- kernels.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel1(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel2(ptr %p) #1 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }
attributes #1 = { "amdgpu-flat-work-group-size"="32,128" "amdgpu-waves-per-eu"="4,6" "amdgpu-max-num-workgroups"="8,8,8" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}
