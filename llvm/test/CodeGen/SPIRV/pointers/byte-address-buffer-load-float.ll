; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; FIXME: spirv-val rejects the generated OpStore for float global sinks (separate
; from ptrcast legalization); legalizer coverage is in SPIRVLegalizePointerCast.ll.

; CHECK-DAG: [[AC:%[0-9]+]] = OpAccessChain {{.*}}
; CHECK-DAG: OpLoad {{.*}} [[AC]]
; CHECK-DAG: OpBitcast
; CHECK-NOT: OpTypeVector
; CHECK: OpStore {{.*}}

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1
@out = addrspace(10) global float zeroinitializer, align 4

define void @main() local_unnamed_addr #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load float, ptr addrspace(11) %ptr, align 4
  store float %val, ptr addrspace(10) @out, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
