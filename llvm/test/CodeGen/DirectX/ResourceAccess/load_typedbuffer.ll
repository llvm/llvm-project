; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @use_float4(<4 x float>)
declare void @use_float(float)

; CHECK-LABEL: define void @load_float4
define void @load_float4(i32 %index, i32 %elemindex) {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)

  ; CHECK: %[[VALUE:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  %vec_data = load <4 x float>, ptr %ptr
  call void @use_float4(<4 x float> %vec_data)

  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VALUE:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: extractelement <4 x float> %[[VALUE]], i32 1
  %y_ptr = getelementptr inbounds <4 x float>, ptr %ptr, i32 0, i32 1
  %y_data = load float, ptr %y_ptr
  call void @use_float(float %y_data)

  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VALUE:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: extractelement <4 x float> %[[VALUE]], i32 %elemindex
  %dynamic = getelementptr inbounds <4 x float>, ptr %ptr, i32 0, i32 %elemindex
  %dyndata = load float, ptr %dynamic
  call void @use_float(float %dyndata)

  ret void
}
