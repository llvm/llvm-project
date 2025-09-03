; RUN: opt -S -disable-output -passes="print<dxil-resources>" < %s 2>&1 | FileCheck %s

@G = external constant <4 x float>, align 4

@Zero.str = private unnamed_addr constant [5 x i8] c"Zero\00", align 1
@One.str = private unnamed_addr constant [4 x i8] c"One\00", align 1
@Two.str = private unnamed_addr constant [4 x i8] c"Two\00", align 1
@Three.str = private unnamed_addr constant [6 x i8] c"Three\00", align 1
@Four.str = private unnamed_addr constant [5 x i8] c"Four\00", align 1
@Array.str = private unnamed_addr constant [6 x i8] c"Array\00", align 1
@Five.str = private unnamed_addr constant [5 x i8] c"Five\00", align 1
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@Constants.str = private unnamed_addr constant [10 x i8] c"Constants\00", align 1

define void @test_typedbuffer() {
  ; ByteAddressBuffer Buf : register(t8, space1)
  %srv0 = call target("dx.RawBuffer", void, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 1, i32 8, i32 1, i32 0, i1 false, ptr @Zero.str)
  ; CHECK: Resource [[SRV0:[0-9]+]]:
  ; CHECK:   Name: Zero
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 1
  ; CHECK:     Lower Bound: 8
  ; CHECK:     Size: 1
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: RawBuffer

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
  %srv1 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 1, i32 0, i1 false, ptr @One.str)
  ; CHECK: Resource [[SRV1:[0-9]+]]:
  ; CHECK:   Name: One
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 1
  ; CHECK:     Space: 4
  ; CHECK:     Lower Bound: 2
  ; CHECK:     Size: 1
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: StructuredBuffer
  ; CHECK:   Buffer Stride: 32
  ; CHECK:   Alignment: 4

  ; Buffer<uint4> Buf[24] : register(t3, space5)
  %srv2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 5, i32 3, i32 24, i32 0, i1 false, ptr @Two.str)
  ; CHECK: Resource [[SRV2:[0-9]+]]:
  ; CHECK:   Name: Two
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 2
  ; CHECK:     Space: 5
  ; CHECK:     Lower Bound: 3
  ; CHECK:     Size: 24
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: Buffer
  ; CHECK:   Element Type: u32
  ; CHECK:   Element Count: 4

  ; RWBuffer<int> Buf : register(u7, space2)
  %uav0 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 2, i32 7, i32 1, i32 0, i1 false, ptr @Three.str)
  ; CHECK: Resource [[UAV0:[0-9]+]]:
  ; CHECK:   Name: Three
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 2
  ; CHECK:     Lower Bound: 7
  ; CHECK:     Size: 1
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Unknown
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: Buffer
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: i32
  ; CHECK:   Element Count: 1

  ; RWBuffer<float4> Buf : register(u5, space3)
  %uav1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 3, i32 5, i32 1, i32 0, i1 false, ptr @Four.str)
  call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav1, i8 -1)
  ; CHECK: Resource [[UAV1:[0-9]+]]:
  ; CHECK:   Name: Four
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 1
  ; CHECK:     Space: 3
  ; CHECK:     Lower Bound: 5
  ; CHECK:     Size: 1
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Decrement
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: Buffer
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  ; RWBuffer<float4> BufferArray[10] : register(u0, space4)
  ; RWBuffer<float4> Buf = BufferArray[0]
  %uav2_1 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
        @llvm.dx.resource.handlefrombinding(i32 4, i32 0, i32 10, i32 0, i1 false, ptr @Array.str)
  ; RWBuffer<float4> Buf = BufferArray[5]
  %uav2_2 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
        @llvm.dx.resource.handlefrombinding(i32 4, i32 0, i32 10, i32 5, i1 false, ptr @Array.str)
  call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav2_2, i8 1)
  ; CHECK: Resource [[UAV2:[0-9]+]]:
  ; CHECK:   Name: Array
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 2
  ; CHECK:     Space: 4
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 10
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Increment
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: Buffer
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  ; RWBuffer<float4> Buf : register(u0, space5)
  %uav3 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 5, i32 0, i32 1, i32 0, i1 false, ptr @Five.str)
  call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav3, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %uav3, i8 1)
  ; CHECK: Resource [[UAV3:[0-9]+]]:
  ; CHECK:   Name: Five
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 3
  ; CHECK:     Space: 5
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 1
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Invalid
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: Buffer
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  %cb0 = call target("dx.CBuffer", {float})
     @llvm.dx.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, i1 false, ptr @CB.str)
  ; CHECK: Resource [[CB0:[0-9]+]]:
  ; CHECK:   Name: CB
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 1
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 1
  ; CHECK:   Class: CBV
  ; CHECK:   Kind: CBuffer
  ; CHECK:   CBuffer size: 4

  %cb1 = call target("dx.CBuffer", target("dx.Layout", {float}, 4, 0))
     @llvm.dx.resource.handlefrombinding(i32 1, i32 8, i32 1, i32 0, i1 false, ptr @Constants.str)
  ; CHECK: Resource [[CB1:[0-9]+]]:
  ; CHECK:   Name: Constants
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 1
  ; CHECK:     Space: 1
  ; CHECK:     Lower Bound: 8
  ; CHECK:     Size: 1
  ; CHECK:   Class: CBV
  ; CHECK:   Kind: CBuffer
  ; CHECK:   CBuffer size: 4

  ; CHECK-NOT: Resource {{[0-9]+}}:

  ret void
}

; CHECK-DAG: Call bound to [[SRV0]]: %srv0 =
; CHECK-DAG: Call bound to [[SRV1]]: %srv1 =
; CHECK-DAG: Call bound to [[SRV2]]: %srv2 =
; CHECK-DAG: Call bound to [[UAV0]]: %uav0 =
; CHECK-DAG: Call bound to [[UAV1]]: %uav1 =
; CHECK-DAG: Call bound to [[UAV2]]: %uav2_1 =
; CHECK-DAG: Call bound to [[UAV2]]: %uav2_2 =
; CHECK-DAG: Call bound to [[CB0]]: %cb0 =
; CHECK-DAG: Call bound to [[CB1]]: %cb1 =

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
