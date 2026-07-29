; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; Pre-`DXILResourceAccess` form: `atomicrmw` still targets a pointer produced
; by `llvm.dx.resource.getpointer`. `ShaderFlagsAnalysis` should still detect
; the 64-bit typed-resource atomic use.

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x08100000
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       64-Bit integer
; CHECK-NEXT: ;       64-bit Atomics on Typed Resources

; CHECK: Function main : 0x08100000
define void @main() #0 {
  %handle = call target("dx.TypedBuffer", i64, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", i64, 1, 0, 0) %handle, i32 0)
  %old = atomicrmw add ptr %ptr, i64 1 monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
