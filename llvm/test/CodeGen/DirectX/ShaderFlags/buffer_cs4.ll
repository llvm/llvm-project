; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel4.0-compute"

@G = external constant <4 x float>, align 4

define void @test_bufferflags() {

  ; ByteAddressBuffer Buf : register(t8, space1)
  %srv0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)

  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; CHECK: ; Shader Flags Value: 0x00020010
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Raw and Structured buffers
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ; D3D11_SB_GLOBAL_FLAG_ENABLE_RAW_AND_STRUCTURED_BUFFERS
; CHECK-NEXT: {{^;$}}
