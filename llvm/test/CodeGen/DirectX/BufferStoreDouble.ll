; RUN: opt -S -dxil-intrinsic-expansion %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define void @storef64(double %0) {
  ; CHECK: [[B:%.*]] = tail call target("dx.TypedBuffer", double, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.TypedBuffer", double, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we split the double and store the lo and hi bits
  ; CHECK: [[SD:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { i32, i32 } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { i32, i32 } [[SD]], 1
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i32> poison, i32 [[Lo]], i32 0
  ; CHECK: [[Vec2:%.*]] = insertelement <2 x i32> [[Vec1]], i32 [[Hi]], i32 1
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_f64_1_0_0t.v2i32(
  ; CHECK-SAME: target("dx.TypedBuffer", double, 1, 0, 0) [[B]], i32 0, <2 x i32> [[Vec2]])
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", double, 1, 0, 0) %buffer, i32 0,
      double %0)
  ret void
}


define void @storev2f64(<2 x double> %0) {
  ; CHECK: [[B:%.*]] = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.TypedBuffer", <2 x double>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v2f64_1_0_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[SD:%.*]] = call { <2 x i32>, <2 x i32> }
  ; CHECK-SAME: @llvm.dx.splitdouble.v2i32(<2 x double> %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 1
  ; CHECK: [[Vec:%.*]] = shufflevector <2 x i32> [[Lo]], <2 x i32> [[Hi]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.typedbuffer.tdx.TypedBuffer_v2f64_1_0_0t.v4i32(
  ; CHECK-SAME: target("dx.TypedBuffer", <2 x double>, 1, 0, 0) [[B]], i32 0, <4 x i32> [[Vec]])
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <2 x double>, 1, 0, 0) %buffer, i32 0,
      <2 x double> %0)
  ret void
}
