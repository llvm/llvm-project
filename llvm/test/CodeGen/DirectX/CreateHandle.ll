; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.0-compute"

declare i32 @some_val();

define void @test_buffers() {
  ; RWBuffer<float4> Buf : register(u5, space3)
  %typed0 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0(
                  i32 3, i32 5, i32 1, i32 4, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 4, i1 false)
  ; CHECK-NOT: @llvm.dx.cast.handle

  ; RWBuffer<int> Buf : register(u7, space2)
  %typed1 = call target("dx.TypedBuffer", i32, 1, 0, 1)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_1_0_1t(
          i32 2, i32 7, i32 1, i32 6, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 6, i1 false)

  ; Buffer<uint4> Buf[24] : register(t3, space5)
  ; Buffer<uint4> typed2 = Buf[4]
  ; Note that the index below is 3 + 4 = 7
  %typed2 = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_i32_0_0_0t(
          i32 5, i32 3, i32 24, i32 7, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 3, i32 7, i1 false)

  ; struct S { float4 a; uint4 b; };
  ; StructuredBuffer<S> Buf : register(t2, space4)
  %struct0 = call target("dx.RawBuffer", {<4 x float>, <4 x i32>}, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_sl_v4f32v4i32s_0_0t(
          i32 4, i32 2, i32 1, i32 10, i1 true)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 2, i32 10, i1 true)

  ; ByteAddressBuffer Buf : register(t8, space1)
  %byteaddr0 = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 12, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 1, i32 12, i1 false)

  ; Buffer<float4> Buf[] : register(t0)
  ; Buffer<float4> typed3 = Buf[ix]
  %typed3_ix = call i32 @some_val()
  %typed3 = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0t(
          i32 0, i32 0, i32 -1, i32 %typed3_ix, i1 false)
  ; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 %typed3_ix, i1 false)

  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
