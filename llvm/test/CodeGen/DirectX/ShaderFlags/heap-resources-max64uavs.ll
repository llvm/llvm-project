; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s -disable-dxil-remove-unused-resources --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; This test verifies that the Max64UAVs flag is *not* set if there
; are 9 or more heap UAVs, since a heap UAVs do not count.
  
target triple = "dxil-pc-shadermodel6.6-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x40000000

; CHECK: Note: shader requires additional functionality:
; CHECK-NOT:  64 UAV slots
; CHECK:      Resource descriptor heap indexing

; CHECK: Function test : 0x40000000
define void @test() "hlsl.export" {
  
  %1 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 0)
  %2 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 1)
  %3 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 2)
  %4 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 3)
  %5 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 4)
  %6 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 5)
  %7 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 6)
  %8 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 7)
  %9 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap(i32 8)
  
  ret void
}

!dx.valver = !{!0}
!0 = !{i32 1, i32 8}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:             Max64UAVs: false
; DXC:             ResourceDescriptorHeapIndexing: true
; DXC:             SamplerDescriptorHeapIndexing: false
; DXC:       NextUnusedBit:   false

