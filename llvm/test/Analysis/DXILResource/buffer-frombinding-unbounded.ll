; RUN: opt -S -disable-output -passes="print<dxil-resources>" %s 2>&1 | FileCheck %s

@One.str = private unnamed_addr constant [4 x i8] c"One\00", align 1
@Two.str = private unnamed_addr constant [4 x i8] c"Two\00", align 1

define void @test_typedbuffer() {
  ; Buffer<uint4> Buf[] : register(t0, space0)
  %srv = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 -1, i32 0, ptr @One.str)
  ; CHECK: Resource [[#SRV:]]:
  ; CHECK:   Name: One
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 0
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 4294967295
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Unknown
  ; CHECK:   Class: SRV
  ; CHECK:   Kind: Buffer
  ; CHECK:   Element Type: u32
  ; CHECK:   Element Count: 4

  ; RWBuffer<float4> BufferArray[4294967262] : register(u0, space0)
  %uav = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
        @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 4294967262, i32 0, ptr @Two.str)
  ; CHECK: Resource [[#UAV:]]:
  ; CHECK:   Name: Two
  ; CHECK:   Binding:
  ; CHECK:     Record ID: 0
  ; CHECK:     Space: 0
  ; CHECK:     Lower Bound: 0
  ; CHECK:     Size: 4294967262
  ; CHECK:   Globally Coherent: 0
  ; CHECK:   Counter Direction: Unknown
  ; CHECK:   Class: UAV
  ; CHECK:   Kind: Buffer
  ; CHECK:   IsROV: 0
  ; CHECK:   Element Type: f32
  ; CHECK:   Element Count: 4

  ret void
}

; CHECK-DAG: Call bound to [[#SRV]]: %srv =
; CHECK-DAG: Call bound to [[#UAV]]: %uav =

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
