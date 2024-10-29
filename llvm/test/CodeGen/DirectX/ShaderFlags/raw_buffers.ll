; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

@G = external constant <4 x float>, align 4

define void @test_bufferflags() {

  ; RWBuffer<int> Buf : register(u7, space2)
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_1_0t(
          i32 2, i32 7, i32 1, i32 0, i1 false)

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
