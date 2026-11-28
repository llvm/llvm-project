; RUN: opt -S -passes=spirv-lower-resource-names %s | FileCheck %s
; RUN: llc -O0 -enable-new-pm -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=SPIRV

; SPIRV-DAG: OpName {{%[0-9]+}} "buf"
; SPIRV-DAG: OpName {{%[0-9]+}} "sampler"

target triple = "spirv1.6-unknown-vulkan1.3-library"

; CHECK-NOT: @.str =
; CHECK-NOT: @.sampler =
; CHECK: @.runtime = private unnamed_addr constant [4 x i8] c"buf\00"

@.str = private unnamed_addr constant [4 x i8] c"buf\00"
@.sampler = private unnamed_addr constant [8 x i8] c"sampler\00"
@.runtime = private unnamed_addr constant [4 x i8] c"buf\00"

; CHECK-LABEL: define void @explicit()
; CHECK: %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata ![[BUF:[0-9]+]])
define void @explicit() {
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, i32 0)
  store i32 1, ptr addrspace(11) %ptr
  ret void
}

; CHECK-LABEL: define target("spirv.Sampler") @implicit()
; CHECK: %handle = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata ![[SAMPLER:[0-9]+]])
define target("spirv.Sampler") @implicit() {
  %handle = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding(i32 0, i32 0, i32 1, i32 0, ptr @.sampler)
  ret target("spirv.Sampler") %handle
}

; IR like this can be encountered at -Od
;
; CHECK-LABEL: define void @loaded_name()
; CHECK-NOT: alloca
; CHECK: %handle = call target("spirv.Image", float, 5, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata ![[BUF]])
; CHECK: %sampler = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata ![[SAMPLER]])
define void @loaded_name() {
  %name.addr = alloca ptr
  %sampler.name.addr = alloca ptr
  store ptr @.str, ptr %name.addr
  store ptr @.sampler, ptr %sampler.name.addr
  %name = load ptr, ptr %name.addr
  %sampler.name = load ptr, ptr %sampler.name.addr
  %handle = call target("spirv.Image", float, 5, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr %name)
  %other = call target("spirv.Image", float, 5, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr %name)
  %sampler = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding(i32 0, i32 0, i32 1, i32 0, ptr %sampler.name)
  ret void
}

; CHECK-LABEL: define void @runtime_use()
; CHECK: %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata ![[BUF]])
; CHECK: load volatile i8, ptr @.runtime
define void @runtime_use() {
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.runtime)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, i32 0)
  %char = load volatile i8, ptr @.runtime
  %value = zext i8 %char to i32
  store i32 %value, ptr addrspace(11) %ptr
  ret void
}

; CHECK-DAG: ![[BUF]] = !{!"buf"}
; CHECK-DAG: ![[SAMPLER]] = !{!"sampler"}
