; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @store_float4
define void @store_float4(<4 x float> %data, i32 %index, i32 %elemindex) {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)

  ; Store the whole value
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ; Store just the .x component
  %scalar = extractelement <4 x float> %data, i32 0
  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[VEC]], float %scalar, i32 0
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index, <4 x float> %[[INSERT]])
  store float %scalar, ptr %ptr

  ; Store just the .y component
  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[VEC]], float %scalar, i32 1
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index, <4 x float> %[[INSERT]])
  %y_ptr = getelementptr inbounds i8, ptr %ptr, i32 4
  store float %scalar, ptr %y_ptr

  ; Store to one of the elements dynamically
  ; CHECK: %[[LOAD:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.typedbuffer.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <4 x float>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[VEC]], float %scalar, i32 %elemindex
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %buffer, i32 %index, <4 x float> %[[INSERT]])
  %dynamic = getelementptr inbounds <4 x float>, ptr %ptr, i32 0, i32 %elemindex
  store float %scalar, ptr %dynamic

  ret void
}

; CHECK-LABEL: define void @store_half4
define void @store_half4(<4 x half> %data, i32 %index) {
  %buffer = call target("dx.TypedBuffer", <4 x half>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f16_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index)

  ; Store the whole value
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f16_1_0_0t.v4f16(target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index, <4 x half> %data)
  store <4 x half> %data, ptr %ptr

  ; Store just the .x component
  %scalar = extractelement <4 x half> %data, i32 0
  ; CHECK: %[[LOAD:.*]] = call { <4 x half>, i1 } @llvm.dx.resource.load.typedbuffer.v4f16.tdx.TypedBuffer_v4f16_1_0_0t(target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <4 x half>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x half> %[[VEC]], half %scalar, i32 0
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f16_1_0_0t.v4f16(target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index, <4 x half> %[[INSERT]])
  store half %scalar, ptr %ptr

  ; Store just the .y component
  ; CHECK: %[[LOAD:.*]] = call { <4 x half>, i1 } @llvm.dx.resource.load.typedbuffer.v4f16.tdx.TypedBuffer_v4f16_1_0_0t(target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <4 x half>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x half> %[[VEC]], half %scalar, i32 1
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v4f16_1_0_0t.v4f16(target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %buffer, i32 %index, <4 x half> %[[INSERT]])
  %y_ptr = getelementptr inbounds i8, ptr %ptr, i32 2
  store half %scalar, ptr %y_ptr

  ret void
}

; CHECK-LABEL: define void @store_double2
define void @store_double2(<2 x double> %data, i32 %index) {
  %buffer = call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index)

  ; Store the whole value
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t.v2f64(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index, <2 x double> %data)
  store <2 x double> %data, ptr %ptr

  ; Store just the .x component
  %scalar = extractelement <2 x double> %data, i32 0
  ; CHECK: %[[LOAD:.*]] = call { <2 x double>, i1 } @llvm.dx.resource.load.typedbuffer.v2f64.tdx.TypedBuffer_v2f64_1_0_0t(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <2 x double>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <2 x double> %[[VEC]], double %scalar, i32 0
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t.v2f64(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index, <2 x double> %[[INSERT]])
  store double %scalar, ptr %ptr

  ; Store just the .y component
  ; CHECK: %[[LOAD:.*]] = call { <2 x double>, i1 } @llvm.dx.resource.load.typedbuffer.v2f64.tdx.TypedBuffer_v2f64_1_0_0t(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index)
  ; CHECK: %[[VEC:.*]] = extractvalue { <2 x double>, i1 } %[[LOAD]], 0
  ; CHECK: %[[INSERT:.*]] = insertelement <2 x double> %[[VEC]], double %scalar, i32 1
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t.v2f64(target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 %index, <2 x double> %[[INSERT]])
  %y_ptr = getelementptr inbounds i8, ptr %ptr, i32 8
  store double %scalar, ptr %y_ptr

  ret void
}
