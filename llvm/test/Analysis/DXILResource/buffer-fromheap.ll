; RUN: opt -S -disable-output -passes="print<dxil-resources>" < %s 2>&1 | FileCheck %s

%struct.S = type { <4 x float>, <4 x i32> }
%P = type <{ float }>
%Q = type <{ <{ [2 x <{ float, target("dx.Padding", 12) }>], float }> }>

; The resources in this test are created in the same order as they appear after
; sorting in the ResourceInfo list because FileCheck cannot match multiline sections
; of text in arbitrary order.

define void @test_typedbuffer() {

  %idx = tail call i32 @llvm.dx.thread.id.in.group(i32 0)

  ; ByteAddressBuffer Buf0 = ResourceDescriptorHeap[5];
  %srv0 = tail call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_i8_0_0t(i32 5)
; CHECK: Resource [[SRV0:0]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: SRV
; CHECK-NEXT:   Kind: RawBuffer

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf1 = ResourceDescriptorHeap[ID.x];
  %srv1 = tail call target("dx.RawBuffer", %struct.S, 0, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_s_struct.Ss_0_0t(i32 %idx)
; CHECK-DAG: Resource [[SRV1:1]]:
; CHECK: HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: SRV
; CHECK-NEXT:   Kind: StructuredBuffer
; CHECK-NEXT:   Buffer Stride: 32
; CHECK-NEXT:   Alignment: 4

  ; Buffer<uint4> Buf2 = ResourceDescriptorHeap[ID.x + 1];
  %add0 = add i32 %idx, 1
  %srv2 = tail call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4i32_0_0_0t(i32 %add0)
; CHECK: Resource [[SRV2:2]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: SRV
; CHECK-NEXT:   Kind: Buffer
; CHECK-NEXT:   Element Type: u32
; CHECK-NEXT:   Element Count: 4


; Make sure this was the last SRV resource in the list.
; CHECK-NOT: Class: SRV

  ; RWStructuredBuffer<double> Buf6 = ResourceDescriptorHeap[ID.x + 5];
  %add5 = add i32 %idx, 5
  %uav0 = tail call target("dx.RawBuffer", double, 1, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_f64_1_0t(i32 %add5)
; CHECK: Resource [[UAV0:3]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: UAV
; CHECK-NEXT:   Kind: StructuredBuffer
; CHECK-NEXT:   IsROV: 0
; CHECK-NEXT:   Buffer Stride: 8
; CHECK-NEXT:   Alignment: 0

  ; RWStructuredBuffer<float4> Buf4 = ResourceDescriptorHeap[ID.x + 3];
  ; Buf4.DecrementCounter();
  %add3 = add i32 %idx, 3
  %uav1 = tail call target("dx.RawBuffer", <4 x float>, 1, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_v4f32_1_0t(i32 %add3)
  %count0 = tail call noundef i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v4f32_1_0t(target("dx.RawBuffer", <4 x float>, 1, 0) %uav1, i8 -1)
; CHECK: Resource [[UAV1:4]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Decrement
; CHECK-NEXT:   Class: UAV
; CHECK-NEXT:   Kind: StructuredBuffer
; CHECK-NEXT:   IsROV: 0
; CHECK-NEXT:   Buffer Stride: 16
; CHECK-NEXT:   Alignment: 0

  ; RWStructuredBuffer<float4> Buf5 = ResourceDescriptorHeap[ID.x + 4];
  ; Buf5.DecrementCounter();
  ; Buf5.IncrementCounter();
  %add4 = add i32 %idx, 4
  %uav2 = tail call target("dx.RawBuffer", <4 x float>, 1, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_v4f32_1_0t(i32 %add4)
  %14 = tail call noundef i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v4f32_1_0t(target("dx.RawBuffer", <4 x float>, 1, 0) %uav2, i8 -1)
  %15 = tail call noundef i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v4f32_1_0t(target("dx.RawBuffer", <4 x float>, 1, 0) %uav2, i8 1)
; CHECK: Resource [[UAV2:5]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Invalid
; CHECK-NEXT:   Class: UAV
; CHECK-NEXT:   Kind: StructuredBuffer
; CHECK-NEXT:   IsROV: 0
; CHECK-NEXT:   Buffer Stride: 16
; CHECK-NEXT:   Alignment: 0


  ; RWBuffer<int> Buf3 = ResourceDescriptorHeap[ID.x + 2];
  %add2 = add i32 %idx, 2
  %uav3 = tail call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_i32_1_0_1t(i32 %add2)
; CHECK: Resource [[UAV3:6]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: UAV
; CHECK-NEXT:   Kind: Buffer
; CHECK-NEXT:   IsROV: 0
; CHECK-NEXT:   Element Type: i32
; CHECK-NEXT:   Element Count: 1

; Make sure this was the last UAV resource in the list.
; CHECK-NOT: Class: UAV

  ; struct P { float a; };
  ; ConstantBuffer<P> CB1 = ResourceDescriptorHeap[ID.x + 6];
  %add6 = add i32 %idx, 6
  %cbv0 = tail call target("dx.CBuffer", %P)
      @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Pst(i32 %add6)
; CHECK: Resource [[CVB0:7]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: CBV
; CHECK-NEXT:   Kind: CBuffer
; CHECK-NEXT:   CBuffer size: 4

  ; struct Q { float b[3]; };
  ; ConstantBuffer<Q> CB2 = ResourceDescriptorHeap[ID.x + 7];
  %add7 = add i32 %idx, 7
  %cvb1 = tail call target("dx.CBuffer", %Q)
      @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Qst(i32 %add7)
; CHECK: Resource [[CVB1:8]]:
; CHECK-NEXT:   HeapIndexID: {{[0-9]+}}
; CHECK-NEXT:   Globally Coherent: 0
; CHECK-NEXT:   Has Atomic64 Use: 0
; CHECK-NEXT:   Counter Direction: Unknown
; CHECK-NEXT:   Class: CBV
; CHECK-NEXT:   Kind: CBuffer
; CHECK-NEXT:   CBuffer size: 36

; Make sure this was the last CBV resource in the list.
; CHECK-NOT: Class: CBV

; Duplicated resources should not be added to the list
; (created from heap with the same index).
  %srv2_dupl = tail call target("dx.RawBuffer", %struct.S, 0, 0)
      @llvm.dx.resource.handlefromheap.tdx.RawBuffer_s_struct.Ss_0_0t(i32 %idx)
  %cvb1_dupl = tail call target("dx.CBuffer", %Q)
      @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Qst(i32 %add7)

  ret void
}

; CHECK-DAG: Call bound to [[SRV0]]:  %srv0 = tail call target("dx.RawBuffer", i8, 0, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_i8_0_0t(i32 5)
; CHECK-DAG: Call bound to [[SRV1]]:  %srv1 = tail call target("dx.RawBuffer", %struct.S, 0, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_s_struct.Ss_0_0t(i32 %idx)
; CHECK-DAG: Call bound to [[SRV2]]:  %srv2 = tail call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0) @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_v4i32_0_0_0t(i32 %add0)
; CHECK-DAG: Call bound to [[UAV0]]:  %uav0 = tail call target("dx.RawBuffer", double, 1, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_f64_1_0t(i32 %add5)
; CHECK-DAG: Call bound to [[UAV1]]:  %uav1 = tail call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_v4f32_1_0t(i32 %add3)
; CHECK-DAG: Call bound to [[UAV2]]:  %uav2 = tail call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_v4f32_1_0t(i32 %add4)
; CHECK-DAG: Call bound to [[UAV3]]:  %uav3 = tail call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefromheap.tdx.TypedBuffer_i32_1_0_1t(i32 %add2)
; CHECK-DAG: Call bound to [[CVB0]]:  %cbv0 = tail call target("dx.CBuffer", %P) @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Pst(i32 %add6)
; CHECK-DAG: Call bound to [[CVB1]]:  %cvb1 = tail call target("dx.CBuffer", %Q) @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Qst(i32 %add7)

; duplicate calls should map to existing resources in the list, not create new ones
; CHECK-DAG: Call bound to [[SRV1]]:  %srv2_dupl = tail call target("dx.RawBuffer", %struct.S, 0, 0) @llvm.dx.resource.handlefromheap.tdx.RawBuffer_s_struct.Ss_0_0t(i32 %idx)
; CHECK-DAG: Call bound to [[CVB1]]:  %cvb1_dupl = tail call target("dx.CBuffer", %Q) @llvm.dx.resource.handlefromheap.tdx.CBuffer_s_Qst(i32 %add7)
