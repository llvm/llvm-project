; RUN: opt -S -passes=dxil-translate-metadata %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@SB.str = private unnamed_addr constant [3 x i8] c"SB\00", align 1

%struct.S = type { <4 x float>, <4 x i32> }

define void @test() {
  ; Buffer<float4>
  %float4 = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @A.str)
  ; CHECK: %"Buffer<float4>" = type { <4 x float> }

  ; Buffer<int>
  %int = call target("dx.TypedBuffer", i32, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  ; CHECK: %"Buffer<int32_t>" = type { i32 }

  ; Buffer<uint3>
  %uint3 = call target("dx.TypedBuffer", <3 x i32>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  ; CHECK: %"Buffer<uint32_t3>" = type { <3 x i32> }

  ; StructuredBuffer<S>
  %struct0 = call target("dx.RawBuffer", %struct.S, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 10, i32 1, i32 0, ptr @SB.str)
  ; CHECK: %"StructuredBuffer<struct.S>" = type { %struct.S }

  ; ByteAddressBuffer
  %byteaddr = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 20, i32 1, i32 0, ptr null)
  ; CHECK: %ByteAddressBuffer = type { i32 }

  ret void
}

; CHECK:      @[[T0:.*]] = external constant %"Buffer<float4>"
; CHECK-NEXT: @[[T1:.*]] = external constant %"Buffer<int32_t>"
; CHECK-NEXT: @[[T2:.*]] = external constant %"Buffer<uint32_t3>"
; CHECK-NEXT: @[[S0:.*]] = external constant %"StructuredBuffer<struct.S>"
; CHECK-NEXT: @[[B0:.*]] = external constant %ByteAddressBuffer

; CHECK: !{i32 0, ptr @[[T0]], !"A"
; CHECK: !{i32 1, ptr @[[T1]], !""
; CHECK: !{i32 2, ptr @[[T2]], !""
; CHECK: !{i32 3, ptr @[[S0]], !"SB"
; CHECK: !{i32 4, ptr @[[B0]], !""

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
