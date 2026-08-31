; Test that a target-dependent waves-per-EU maximum remains absent during
; ThinLTO propagation. The AMDGPU backend supplies the gfx950 maximum.
;
; Use O0 so the pre-codegen IR directly shows the attribute applied by the
; ThinLTO backend callback.
;
; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/kernel.ll -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %t/device.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.s -O0 -mcpu=gfx950 -filetype=asm \
; RUN:   -amdgpu-enable-object-linking \
; RUN:   -r %t1.bc,kernel,px \
; RUN:   -r %t1.bc,device_func,l \
; RUN:   -r %t2.bc,device_func,px \
; RUN:   -save-temps
; RUN: llvm-dis %t.s.2.5.precodegen.bc -o - | FileCheck %s
; RUN: FileCheck %s --check-prefix=ASM --input-file=%t.s.2

; CHECK: define void @device_func({{.*}}){{.*}} #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {
; CHECK-SAME: "amdgpu-waves-per-eu"="4"

; ASM: .amdgpu_occupancy 4

;--- kernel.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @device_func(ptr)

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="4" }

;--- device.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @device_func(ptr %p) {
  store volatile i32 42, ptr %p
  ret void
}
