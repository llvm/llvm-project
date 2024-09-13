; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @storefloat(<4 x float> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, float [[DATA0_0]], float [[DATA0_1]], float [[DATA0_2]], float [[DATA0_3]], i8 15)
  call void @llvm.dx.typedBufferStore(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer,
      i32 %index, <4 x float> %data)

  ret void
}

define void @storeint(<4 x i32> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i32>, 1, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4i32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x i32> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x i32> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x i32> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x i32> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, i32 [[DATA0_0]], i32 [[DATA0_1]], i32 [[DATA0_2]], i32 [[DATA0_3]], i8 15)
  call void @llvm.dx.typedBufferStore(
      target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) %buffer,
      i32 %index, <4 x i32> %data)

  ret void
}

define void @storehalf(<4 x half> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x half>, 1, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f16_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x half> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x half> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x half> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x half> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.f16(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, half [[DATA0_0]], half [[DATA0_1]], half [[DATA0_2]], half [[DATA0_3]], i8 15)
  call void @llvm.dx.typedBufferStore(
      target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer,
      i32 %index, <4 x half> %data)

  ret void
}

define void @storei16(<4 x i16> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 217, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i16>, 1, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4i16_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x i16> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x i16> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x i16> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x i16> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.i16(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, i16 [[DATA0_0]], i16 [[DATA0_1]], i16 [[DATA0_2]], i16 [[DATA0_3]], i8 15)
  call void @llvm.dx.typedBufferStore(
      target("dx.TypedBuffer", <4 x i16>, 1, 0, 0) %buffer,
      i32 %index, <4 x i16> %data)

  ret void
}
