; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @scalar_user(float)
declare void @vector_user(<4 x float>)

define void @loadfloats() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call <4 x float> @llvm.dx.typedBufferLoad(
             target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 0)

  ; The extract order depends on the users, so don't enforce that here.
  ; CHECK-DAG: extractvalue %dx.types.ResRet.f32 [[DATA0]], 0
  %data0_0 = extractelement <4 x float> %data0, i32 0
  ; CHECK-DAG: extractvalue %dx.types.ResRet.f32 [[DATA0]], 2
  %data0_2 = extractelement <4 x float> %data0, i32 2

  ; If all of the uses are extracts, we skip creating a vector
  ; CHECK-NOT: insertelement
  call void @scalar_user(float %data0_0)
  call void @scalar_user(float %data0_2)

  ; CHECK: [[DATA4:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle [[HANDLE]], i32 4, i32 undef)
  %data4 = call <4 x float> @llvm.dx.typedBufferLoad(
             target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 4)

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 3
  ; CHECK: insertelement <4 x float> undef
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  call void @vector_user(<4 x float> %data4)

  ; CHECK: [[DATA12:%.*]] = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle [[HANDLE]], i32 12, i32 undef)
  %data12 = call <4 x float> @llvm.dx.typedBufferLoad(
             target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i32 12)

  ; CHECK: [[DATA12_3:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA12]], 3
  %data12_3 = extractelement <4 x float> %data12, i32 3

  ; If there are a mix of users we need the vector, but extracts are direct
  ; CHECK: call void @scalar_user(float [[DATA12_3]])
  call void @scalar_user(float %data12_3)
  call void @vector_user(<4 x float> %data12)

  ret void
}

define void @loadint() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 0)
              @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4i32_0_0_0(
                  i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle [[HANDLE]], i32 0, i32 undef)
  %data0 = call <4 x i32> @llvm.dx.typedBufferLoad(
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
  %data0 = call <4 x half> @llvm.dx.typedBufferLoad(
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
  %data0 = call <4 x i16> @llvm.dx.typedBufferLoad(
             target("dx.TypedBuffer", <4 x i16>, 0, 0, 0) %buffer, i32 0)

  ret void
}
