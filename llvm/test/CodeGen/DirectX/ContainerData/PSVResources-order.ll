; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s

; Check that resources are emitted to the object in the order that matches what
; the DXIL validator expects: CBuffers, Samplers, SRVs, and then UAVs.

; CHECK: Resources:
; CHECK: - Type: CBV
; TODO:  - Type: Sampler
; CHECK: - Type: SRVRaw
; CHECK: - Type: UAVTyped

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(
          i32 2, i32 7, i32 1, i32 0, ptr null)
  %srv0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, ptr null)
  %cbuf = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
      @llvm.dx.resource.handlefrombinding(i32 3, i32 2, i32 1, i32 0, ptr null)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
