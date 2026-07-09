; spirv-legalize-pointer-cast consumes spv.ptrcast intrinsics produced by
; spirv-emit-intrinsics, so we chain both passes and check the ptrcast is
; rewritten into a sequence of typed loads + gep/extractelt.
;
; RUN: opt -S -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

@M = internal addrspace(10) global [4 x <2 x float>] zeroinitializer, align 4
@OUT = internal addrspace(10) global float zeroinitializer, align 4

; Loading a <5 x float> through a [4 x <2 x float>] forces emit-intrinsics to
; insert spv.ptrcast; legalize-pointer-cast lowers it to typed <2 x float>
; loads stitched together with extractelt. After the pass, no spv.ptrcast call
; should remain.

define spir_func void @main() #0 {
; CHECK-LABEL: define spir_func void @main(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: call ptr addrspace(10) {{.*}}@llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M, i32 0, i32 0)
; CHECK: load <2 x float>, ptr addrspace(10)
; CHECK: call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float>
entry:
  %v = load <5 x float>, ptr addrspace(10) @M, align 4
  %x = extractelement <5 x float> %v, i32 4
  store float %x, ptr addrspace(10) @OUT, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
