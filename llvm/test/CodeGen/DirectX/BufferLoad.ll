; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @scalar_user(float)

define void @loadfloats() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call { float, float, float, float } @llvm.dx.typedBufferLoad.f32(
             target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 0)

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA0]], 0
  %data0_0 = extractvalue {float, float, float, float} %data0, 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA0]], 2
  %data0_2 = extractvalue {float, float, float, float} %data0, 2

  call void @scalar_user(float %data0_0)
  call void @scalar_user(float %data0_2)

  ret void
}

define void @loadint() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4i32_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call {i32, i32, i32, i32} @llvm.dx.typedBufferLoad.i32(
             target("dx.TypedBuffer", <4 x i32>, 0, 0, 0) %buffer, i32 0)

  ret void
}

define void @loadhalf() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x half>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f16_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.f16 @dx.op.bufferLoad.f16(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call {half, half, half, half} @llvm.dx.typedBufferLoad.f16(
             target("dx.TypedBuffer", <4 x half>, 0, 0, 0) %buffer, i32 0)

  ret void
}

define void @loadi16() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i16>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4i16_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.i16 @dx.op.bufferLoad.i16(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call {i16, i16, i16, i16} @llvm.dx.typedBufferLoad.i16(
             target("dx.TypedBuffer", <4 x i16>, 0, 0, 0) %buffer, i32 0)

  ret void
}
