; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

@G = external constant <4 x float>, align 4

define void @test_bufferflags() {

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
  %struct0 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 10, i1 true)

; CHECK: ; Shader Flags Value: 0x00020000
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Raw and Structured buffers
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: {{^;$}}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC-NEXT:       Doubles: false
; DXC-NEXT:       ComputeShadersPlusRawAndStructuredBuffers:         true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:       NextUnusedBit:   false
; DXC: ...

  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
