; Modified from: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_usm_storage_classes/intel_usm_addrspaces.ll

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_usm_storage_classes %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_usm_storage_classes %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-WITHOUT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-: Capability USMStorageClassesINTEL
; CHECK-SPIRV-WITHOUT-NO: Capability USMStorageClassesINTEL
; CHECK-SPIRV-EXT-DAG: %[[DevTy:[0-9]+]] = OpTypePointer DeviceOnlyINTEL %[[#]]
; CHECK-SPIRV-EXT-DAG: %[[HostTy:[0-9]+]] = OpTypePointer HostOnlyINTEL %[[#]]
; CHECK-SPIRV-DAG: %[[CrsWrkTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @foo_kernel() {
entry:
  ret void
}

; CHECK-SPIRV: %[[Ptr1:[0-9]+]] = OpLoad %[[CrsWrkTy]] %[[#]]
; CHECK-SPIRV-EXT: %[[CastedPtr1:[0-9]+]] = OpCrossWorkgroupCastToPtrINTEL %[[DevTy]] %[[Ptr1]]
; CHECK-SPIRV-WITHOUT-NOT: OpCrossWorkgroupCastToPtrINTEL
; CHECK-SPIRV-EXT: OpStore %[[#]] %[[CastedPtr1]]
define spir_func void @test1(ptr addrspace(1) %arg_glob, ptr addrspace(5) %arg_dev) {
entry:
  %arg_glob.addr = alloca ptr addrspace(1), align 4
  %arg_dev.addr = alloca ptr addrspace(5), align 4
  store ptr addrspace(1) %arg_glob, ptr %arg_glob.addr, align 4
  store ptr addrspace(5) %arg_dev, ptr %arg_dev.addr, align 4
  %loaded_glob = load ptr addrspace(1), ptr %arg_glob.addr, align 4
  %casted_ptr = addrspacecast ptr addrspace(1) %loaded_glob to ptr addrspace(5)
  store ptr addrspace(5) %casted_ptr, ptr %arg_dev.addr, align 4
  ret void
}

; CHECK-SPIRV: %[[Ptr2:[0-9]+]] = OpLoad %[[CrsWrkTy]] %[[#]]
; CHECK-SPIRV-EXT: %[[CastedPtr2:[0-9]+]] = OpCrossWorkgroupCastToPtrINTEL %[[HostTy]] %[[Ptr2]]
; CHECK-SPIRV-WITHOUT-NOT: OpCrossWorkgroupCastToPtrINTEL
; CHECK-SPIRV-EXT: OpStore %[[#]] %[[CastedPtr2]]
define spir_func void @test2(ptr addrspace(1) %arg_glob, ptr addrspace(6) %arg_host) {
entry:
  %arg_glob.addr = alloca ptr addrspace(1), align 4
  %arg_host.addr = alloca ptr addrspace(6), align 4
  store ptr addrspace(1) %arg_glob, ptr %arg_glob.addr, align 4
  store ptr addrspace(6) %arg_host, ptr %arg_host.addr, align 4
  %loaded_glob = load ptr addrspace(1), ptr %arg_glob.addr, align 4
  %casted_ptr = addrspacecast ptr addrspace(1) %loaded_glob to ptr addrspace(6)
  store ptr addrspace(6) %casted_ptr, ptr %arg_host.addr, align 4
  ret void
}

; CHECK-SPIRV-EXT: %[[Ptr3:[0-9]+]] = OpLoad %[[DevTy]] %[[#]]
; CHECK-SPIRV-EXT: %[[CastedPtr3:[0-9]+]] = OpPtrCastToCrossWorkgroupINTEL %[[CrsWrkTy]] %[[Ptr3]]
; CHECK-SPIRV-WITHOUT-NOT: OpPtrCastToCrossWorkgroupINTEL
; CHECK-SPIRV-EXT: OpStore %[[#]] %[[CastedPtr3]]
define spir_func void @test3(ptr addrspace(1) %arg_glob, ptr addrspace(5) %arg_dev) {
entry:
  %arg_glob.addr = alloca ptr addrspace(1), align 4
  %arg_dev.addr = alloca ptr addrspace(5), align 4
  store ptr addrspace(1) %arg_glob, ptr %arg_glob.addr, align 4
  store ptr addrspace(5) %arg_dev, ptr %arg_dev.addr, align 4
  %loaded_dev = load ptr addrspace(5), ptr %arg_dev.addr, align 4
  %casted_ptr = addrspacecast ptr addrspace(5) %loaded_dev to ptr addrspace(1)
  store ptr addrspace(1) %casted_ptr, ptr %arg_glob.addr, align 4
  ret void
}

; CHECK-SPIRV-EXT: %[[Ptr4:[0-9]+]] = OpLoad %[[HostTy]] %[[#]]
; CHECK-SPIRV-EXT: %[[CastedPtr4:[0-9]+]] = OpPtrCastToCrossWorkgroupINTEL %[[CrsWrkTy]] %[[Ptr4]]
; CHECK-SPIRV-WITHOUT-NOT: OpPtrCastToCrossWorkgroupINTEL
; CHECK-SPIRV-EXT: OpStore %[[#]] %[[CastedPtr4]]
define spir_func void @test4(ptr addrspace(1) %arg_glob, ptr addrspace(6) %arg_host) {
entry:
  %arg_glob.addr = alloca ptr addrspace(1), align 4
  %arg_host.addr = alloca ptr addrspace(6), align 4
  store ptr addrspace(1) %arg_glob, ptr %arg_glob.addr, align 4
  store ptr addrspace(6) %arg_host, ptr %arg_host.addr, align 4
  %loaded_host = load ptr addrspace(6), ptr %arg_host.addr, align 4
  %casted_ptr = addrspacecast ptr addrspace(6) %loaded_host to ptr addrspace(1)
  store ptr addrspace(1) %casted_ptr, ptr %arg_glob.addr, align 4
  ret void
}
