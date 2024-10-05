; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @scalar_user(float)
declare void @vector_user(<4 x float>)
declare void @check_user(i1)

declare void @vector_user_v3f32x4(<12 x float>)

;; StructureBuffer load

define void @loadv4f32() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", <4 x float>, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v4f32_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 0, i32 0, i8 15, i32 4)
  %data0 = call <4 x float> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <4 x float>, 0, 0) %buffer, i32 0, i32 0)

  ; The extract order depends on the users, so don't enforce that here.
  ; CHECK-DAG: [[VAL0_0:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA0]], 0
  %data0_0 = extractelement <4 x float> %data0, i32 0
  ; CHECK-DAG: [[VAL0_2:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA0]], 2
  %data0_2 = extractelement <4 x float> %data0, i32 2

  ; If all of the uses are extracts, we skip creating a vector
  ; CHECK-NOT: insertelement
  ; CHECK-DAG: call void @scalar_user(float [[VAL0_0]])
  ; CHECK-DAG: call void @scalar_user(float [[VAL0_2]])
  call void @scalar_user(float %data0_0)
  call void @scalar_user(float %data0_2)

  ; CHECK: [[DATA4:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 4, i32 0, i8 15, i32 4)
  %data4 = call <4 x float> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <4 x float>, 0, 0) %buffer, i32 4, i32 0)

  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 0
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 1
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 2
  ; CHECK: extractvalue %dx.types.ResRet.f32 [[DATA4]], 3
  ; CHECK: insertelement <4 x float> undef
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  ; CHECK: insertelement <4 x float>
  call void @vector_user(<4 x float> %data4)

  ; CHECK: [[DATA12:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 12, i32 0, i8 15, i32 4)
  %data12 = call <4 x float> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <4 x float>, 0, 0) %buffer, i32 12, i32 0)

  ; CHECK: [[DATA12_3:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA12]], 3
  %data12_3 = extractelement <4 x float> %data12, i32 3

  ; If there are a mix of users we need the vector, but extracts are direct
  ; CHECK: call void @scalar_user(float [[DATA12_3]])
  call void @scalar_user(float %data12_3)
  call void @vector_user(<4 x float> %data12)

  ret void
}

define void @loadv3f32x4() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", <12 x float>, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v3f32x4_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; The temporary casts should all have been cleaned up
  ; CHECK-NOT: %dx.cast_handle

  ; CHECK: [[DATA0_3:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 0, i32 0, i8 15, i32 4)
  ; CHECK: [[DATA4_7:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 0, i32 16, i8 15, i32 4)
  ; CHECK: [[DATA8_11:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle [[HANDLE]], i32 0, i32 32, i8 15, i32 4)
  %data0 = call <12 x float> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <12 x float>, 0, 0) %buffer, i32 0, i32 0)

  ; The extract order depends on the users, so don't enforce that here.
  ; CHECK-DAG: [[VAL0_2:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA0_3]], 2
  %data0_2 = extractelement <12 x float> %data0, i32 2
  ; CHECK-DAG: [[VAL0_7:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA4_7]], 3
  %data0_7 = extractelement <12 x float> %data0, i32 7

  ; If all of the uses are extracts, we skip creating a vector
  ; CHECK-NOT: insertelement
  ; CHECK-DAG: call void @scalar_user(float [[VAL0_2]])
  ; CHECK-DAG: call void @scalar_user(float [[VAL0_7]])
  call void @scalar_user(float %data0_2)
  call void @scalar_user(float %data0_7)

  ;; Vector Use
  ;
  ; CHECK: [[DATA3_0_3:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %buffer_annot, i32 3, i32 0, i8 15, i32 4)
  ; CHECK: [[DATA3_4_7:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %buffer_annot, i32 3, i32 16, i8 15, i32 4)
  ; CHECK: [[DATA3_8_11:%.*]] = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %buffer_annot, i32 3, i32 32, i8 15, i32 4)
  ; CHECK: [[VAL3_0:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_0_3]], 0
  ; CHECK: [[VAL3_1:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_0_3]], 1
  ; CHECK: [[VAL3_2:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_0_3]], 2
  ; CHECK: [[VAL3_3:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_0_3]], 3
  ; CHECK: [[VAL3_4:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_4_7]], 0
  ; CHECK: [[VAL3_5:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_4_7]], 1
  ; CHECK: [[VAL3_6:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_4_7]], 2
  ; CHECK: [[VAL3_7:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_4_7]], 3
  ; CHECK: [[VAL3_8:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_8_11]], 0
  ; CHECK: [[VAL3_9:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_8_11]], 1
  ; CHECK: [[VAL3_10:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_8_11]], 2
  ; CHECK: [[VAL3_11:%.*]] = extractvalue %dx.types.ResRet.f32 [[DATA3_8_11]], 3
  ; CHECK: insertelement <12 x float> undef
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: insertelement <12 x float>
  ; CHECK: [[VAL3_VecRes:%.*]] = insertelement <12 x float>
  ; CHECK: call void @vector_user_v3f32x4(<12 x float> [[VAL3_VecRes]])
  %data3 = call <12 x float> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <12 x float>, 0, 0) %buffer, i32 3, i32 0)
  call void @vector_user_v3f32x4(<12 x float> %data3);

  ret void
}

define void @loadv4i32x2() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", <8 x i32>, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v4i32x2_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0_3:%.*]] = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 0, i8 15, i32 4)
  ; CHECK: [[DATA4_7:%.*]] = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 16, i8 15, i32 4)
  %data0 = call <8 x i32> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <8 x i32>, 0, 0) %buffer, i32 0, i32 0)

  ret void
}

define void @loadv4f16() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", <4 x half>, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v4f16_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0:%.*]] = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 0, i8 15, i32 2)
  %data0 = call <4 x half> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <4 x half>, 0, 0) %buffer, i32 0, i32 0)

  ret void
}

define void @loadv2i16x3() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", <6 x i16>, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v2i16x3_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0_3:%.*]] = call %dx.types.ResRet.i16 @dx.op.rawBufferLoad.i16(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 0, i8 15, i32 2)
  ; CHECK: [[DATA4_5:%.*]] = call %dx.types.ResRet.i16 @dx.op.rawBufferLoad.i16(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 8, i8 3, i32 2)
  %data0 = call <6 x i16> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", <6 x i16>, 0, 0) %buffer, i32 0, i32 0)

  ret void
}

;; ByteAddressBuffer load
define void @load_2() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_v2i32_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK: [[DATA0_1:%.*]] = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %buffer_annot, i32 0, i32 0, i8 3, i32 4)
  %data0 = call <2 x i32> @llvm.dx.rawBufferLoad(
      target("dx.RawBuffer", i8, 0, 0) %buffer, i32 0, i32 0)

  ret void
}
