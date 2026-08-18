; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s -disable-dxil-remove-unused-resources --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; This test makes sure that the shader flag 'Resource descriptor heap indexing'
; is set when the shader uses CreateHandleFromHeap instruction to get a resource
; from a resource descriptor heap, and that the shader flag `Sampler descriptor
; heap indexing` is set when the shader uses the same instruction to get a sampler
; from a sampler descriptor heap.

target triple = "dxil-pc-shadermodel6.6-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0xc0000000

; CHECK: Note: shader requires additional functionality:
; CHECK:        Resource descriptor heap indexing
; CHECK:        Sampler descriptor heap indexing
;
; CHECK: Function test_1 : 0x40000000
define void @test_1() "hlsl.export" {
  ; RWBuffer<float4> Buf = ResourceDescriptorHeap[3]
  %typed = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4f32_1_0_0(i32 3, i1 false)
  ret void
}

; CHECK: Function test_2 : 0x80000000
define void @test_2() "hlsl.export" {
  ; SamplerState Samp = SamplerDescriptorHeap[100];
  %samp = call target("dx.Sampler", 0)
              @llvm.dx.resource.handlefromheap.tdx.Sampler_0(i32 100, i1 true)
  ret void
}

!dx.valver = !{!0}
!0 = !{i32 1, i32 8}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:             ResourceDescriptorHeapIndexing: true
; DXC:             SamplerDescriptorHeapIndexing: true
; DXC:       NextUnusedBit:   false

