; Test cross-TU occupancy attribute propagation via ThinLTO.
; A kernel in one module calls a device function in another module.
; The device function should receive the kernel's occupancy attributes.
;
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/kernel.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/device.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o \
; RUN:   -r %t1.bc,kernel,px \
; RUN:   -r %t1.bc,device_func,l \
; RUN:   -r %t2.bc,device_func,px \
; RUN:   -save-temps
; RUN: llvm-dis %t.o.2.5.precodegen.bc -o - | FileCheck %s

; CHECK: define void @device_func({{.*}}) {{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {
; CHECK-SAME: "amdgpu-flat-work-group-size"="64,256"
; CHECK-SAME: "amdgpu-max-num-workgroups"="16,16,1"
; CHECK-SAME: "amdgpu-waves-per-eu"="2,8"

;--- kernel.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}
