; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

@G = external constant <4 x float>, align 4

define void @test_bufferflags() {

  ; struct S { float4 a; uint4 b; };
  ; RWStructuredBuffer<S> Buf : register(u2, space4)
  %struct0 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 1, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 10, i1 true)

  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; CHECK: ; Shader Flags Value: 0x00000010
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ; D3D11_SB_GLOBAL_FLAG_ENABLE_RAW_AND_STRUCTURED_BUFFERS
; CHECK-NEXT: {{^;$}}
