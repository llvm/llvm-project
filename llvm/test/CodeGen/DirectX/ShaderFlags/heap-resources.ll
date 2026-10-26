; RUN: opt -S --passes="print-dx-shader-flags" -mtriple=dxil-pc-shadermodel6.6-library 2>&1 %s | FileCheck %s --check-prefixes=CHECK,CHECK-66
; RUN: opt -S --passes="print-dx-shader-flags" -mtriple=dxil-pc-shadermodel6.7-library 2>&1 %s | FileCheck %s --check-prefixes=CHECK,CHECK-67
; RUN: llc %s -disable-dxil-remove-unused-resources -mtriple=dxil-pc-shadermodel6.7-library --filetype=obj -o - | \
; RUN:   obj2yaml | FileCheck %s --check-prefix=DXC

; This test makes sure that the shader flag 'Resource descriptor heap indexing'
; is set when the shader uses llvm.dx.resource.handlefromheap intrinsic to get
; a resource from a resource descriptor heap, and that the shader flag 
; `Sampler descriptor heap indexing` is set when the shader uses the same
; intrinsic to get a sampler from a sampler descriptor heap.

; It also checks that the flag "Any UAV may not alias any other UAV" is set for
; a function that uses a heap UAV resource, but only for shader model 6.7 and higher. 

; CHECK:      Combined Shader Flags for Module
; CHECK-66:   Shader Flags Value: 0xc0000000
; CHECK-67:   Shader Flags Value: 0x2c0000000

; CHECK: Note: shader requires additional functionality:
; CHECK:        Resource descriptor heap indexing
; CHECK:        Sampler descriptor heap indexing

; CHECK: Note: extra DXIL module flags:
; CHECK-67:     Any UAV may not alias any other UAV
;
; CHECK-66: Function test_1 : 0x40000000
; CHECK-67: Function test_1 : 0x240000000
define void @test_1() "hlsl.export" {
  ; RWBuffer<float4> Buf = ResourceDescriptorHeap[3]
  %typed = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4f32_1_0_0(i32 3)
  ret void
}

; CHECK: Function test_2 : 0x80000000
define void @test_2() "hlsl.export" {
  ; SamplerState Samp = SamplerDescriptorHeap[100];
  %samp = call target("dx.Sampler", 0)
              @llvm.dx.resource.handlefromheap.tdx.Sampler_0(i32 100)
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

