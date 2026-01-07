; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @storefloats
define void @storefloats(<4 x float> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, float [[DATA0_0]], float [[DATA0_1]], float [[DATA0_2]], float [[DATA0_3]], i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer,
      i32 %index, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @storeonefloat
define void @storeonefloat(float %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, float %data, float %data, float %data, float %data, i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", float, 1, 0, 0) %buffer,
      i32 %index, float %data)

  ret void
}

; CHECK-LABEL: define void @storetwofloat
define void @storetwofloat(<2 x float> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <2 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <2 x float> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <2 x float> %data, i32 1
  ; CHECK: call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, float [[DATA0_0]], float [[DATA0_1]], float [[DATA0_0]], float [[DATA0_0]], i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <2 x float>, 1, 0, 0) %buffer,
      i32 %index, <2 x float> %data)

  ret void
}

; CHECK-LABEL: define void @storeint
define void @storeint(<4 x i32> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i32>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4i32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x i32> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x i32> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x i32> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x i32> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, i32 [[DATA0_0]], i32 [[DATA0_1]], i32 [[DATA0_2]], i32 [[DATA0_3]], i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) %buffer,
      i32 %index, <4 x i32> %data)

  ret void
}

; CHECK-LABEL: define void @storehalf
define void @storehalf(<4 x half> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x half>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f16_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x half> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x half> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x half> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x half> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.f16(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, half [[DATA0_0]], half [[DATA0_1]], half [[DATA0_2]], half [[DATA0_3]], i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer,
      i32 %index, <4 x half> %data)

  ret void
}

; CHECK-LABEL: define void @storei16
define void @storei16(<4 x i16> %data, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x i16>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4i16_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[DATA0_0:%.*]] = extractelement <4 x i16> %data, i32 0
  ; CHECK: [[DATA0_1:%.*]] = extractelement <4 x i16> %data, i32 1
  ; CHECK: [[DATA0_2:%.*]] = extractelement <4 x i16> %data, i32 2
  ; CHECK: [[DATA0_3:%.*]] = extractelement <4 x i16> %data, i32 3
  ; CHECK: call void @dx.op.bufferStore.i16(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, i16 [[DATA0_0]], i16 [[DATA0_1]], i16 [[DATA0_2]], i16 [[DATA0_3]], i8 15){{$}}
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x i16>, 1, 0, 0) %buffer,
      i32 %index, <4 x i16> %data)

  ret void
}

; CHECK-LABEL: define void @store_scalarized_floats
define void @store_scalarized_floats(float %data0, float %data1, float %data2, float %data3, i32 %index) {

  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; We shouldn't end up with any inserts/extracts.
  ; CHECK-NOT: insertelement
  ; CHECK-NOT: extractelement

  ; CHECK: call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle [[HANDLE]], i32 %index, i32 undef, float %data0, float %data1, float %data2, float %data3, i8 15)
  %vec.upto0 = insertelement <4 x float> poison, float %data0, i64 0
  %vec.upto1 = insertelement <4 x float> %vec.upto0, float %data1, i64 1
  %vec.upto2 = insertelement <4 x float> %vec.upto1, float %data2, i64 2
  %vec = insertelement <4 x float> %vec.upto2, float %data3, i64 3
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer,
      i32 %index, <4 x float> %vec)

  ret void
}

define void @storef64(<2 x i32> %0) {
  ; CHECK: [[B1:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[BA:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[B1]]
  
  %buffer = tail call target("dx.TypedBuffer", double, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[D0:%.*]] = extractelement <2 x i32> %0, i32 0
  ; CHECK: [[D1:%.*]] = extractelement <2 x i32> %0, i32 1
  ; CHECK: call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle [[BA]], i32 0, i32 undef, i32 %2, i32 %3, i32 %2, i32 %2, i8 15)
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0, <2 x i32> %0)
  ret void
}

define void @storev2f64(<4 x i32> %0) {
  ; CHECK: [[B1:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217,
  ; CHECK: [[BA:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[B1]]
  
  %buffer = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.resource.casthandle

  ; CHECK: [[D0:%.*]] = extractelement <4 x i32> %0, i32 0
  ; CHECK: [[D1:%.*]] = extractelement <4 x i32> %0, i32 1
  ; CHECK: [[D2:%.*]] = extractelement <4 x i32> %0, i32 2
  ; CHECK: [[D3:%.*]] = extractelement <4 x i32> %0, i32 3
  ; CHECK: call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle [[BA]], i32 0, i32 undef, i32 [[D0]], i32 [[D1]], i32 [[D2]], i32 [[D3]], i8 15)
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 0,
      <4 x i32> %0)
  ret void
}
