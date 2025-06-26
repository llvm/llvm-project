; RUN: opt -S -dxil-intrinsic-expansion %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.2-compute"

define void @storef64(double %0, i32 %index) {
  ; CHECK: [[B:%.*]] = tail call target("dx.RawBuffer", double, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.RawBuffer", double, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f64_1_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; check we split the double and store the lo and hi bits
  ; CHECK: [[SD:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { i32, i32 } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { i32, i32 } [[SD]], 1
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i32> poison, i32 [[Lo]], i32 0
  ; CHECK: [[Vec2:%.*]] = insertelement <2 x i32> [[Vec1]], i32 [[Hi]], i32 1
  ; this shufflevector is unnecessary but generated to avoid specalization
  ; CHECK: [[Vec3:%.*]] = shufflevector <2 x i32> [[Vec2]], <2 x i32> poison, <2 x i32> <i32 0, i32 1>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_f64_1_0t.v2i32(
  ; CHECK-SAME: target("dx.RawBuffer", double, 1, 0) [[B]], i32 %index, i32 0, <2 x i32> [[Vec3]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", double, 1, 0) %buffer, i32 %index, i32 0,
      double %0)
  ret void
}

define void @storev2f64(<2 x double> %0, i32 %index) {
  ; CHECK: [[B:%.*]] = tail call target("dx.RawBuffer", <2 x double>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v2f64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.RawBuffer", <2 x double>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v2f64_1_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; CHECK: [[SD:%.*]] = call { <2 x i32>, <2 x i32> }
  ; CHECK-SAME: @llvm.dx.splitdouble.v2i32(<2 x double> %0)
  ; CHECK: [[Lo:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 1
  ; CHECK: [[Vec:%.*]] = shufflevector <2 x i32> [[Lo]], <2 x i32> [[Hi]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; this shufflevector is unnecessary but generated to avoid specalization
  ; CHECK: [[Vec2:%.*]] = shufflevector <4 x i32> [[Vec]], <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v2f64_1_0t.v4i32(
  ; CHECK-SAME: target("dx.RawBuffer", <2 x double>, 1, 0) [[B]], i32 %index, i32 0, <4 x i32> [[Vec2]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <2 x double>, 1, 0) %buffer, i32 %index, i32 0,
      <2 x double> %0)
  ret void
}

define void @storev3f64(<3 x double> %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", <3 x double>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v3f64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.RawBuffer", <3 x double>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v3f64_1_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; CHECK: [[A:%.*]] = call { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double> %0)
  ; CHECK: [[B:%.*]] = extractvalue { <3 x i32>, <3 x i32> } [[A]], 0
  ; CHECK: [[C:%.*]] = extractvalue { <3 x i32>, <3 x i32> } [[A]], 1
  ; CHECK: [[D:%.*]] = shufflevector <3 x i32> [[B]], <3 x i32> [[C]], <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
  ; CHECK: [[E:%.*]] = shufflevector <6 x i32> [[D]], <6 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v3f64_1_0t.v4i32(target("dx.RawBuffer", <3 x double>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i32> [[E]])
  ; CHECK: [[F:%.*]] = shufflevector <6 x i32> [[D]], <6 x i32> poison, <2 x i32> <i32 4, i32 5>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v3f64_1_0t.v2i32(target("dx.RawBuffer", <3 x double>, 1, 0) [[Buf]], i32 %index, i32 16, <2 x i32> [[F]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <3 x double>, 1, 0) %buffer, i32 %index, i32 0,
      <3 x double> %0)
  ret void
}

define void @storev4f64(<4 x double> %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", <4 x double>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %buffer = tail call target("dx.RawBuffer", <4 x double>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f64_1_0t(
          i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)

  ; CHECK: [[A:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.dx.splitdouble.v4i32(<4 x double> %0)
  ; CHECK: [[B:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[A]], 0
  ; CHECK: [[C:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[A]], 1
  ; CHECK: [[D:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[C]], <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  ; CHECK: [[E:%.*]] = shufflevector <8 x i32> [[D]], <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f64_1_0t.v4i32(target("dx.RawBuffer", <4 x double>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i32> [[E]])
  ; CHECK: [[F:%.*]] = shufflevector <8 x i32> [[D]], <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f64_1_0t.v4i32(target("dx.RawBuffer", <4 x double>, 1, 0) [[Buf]], i32 %index, i32 16, <4 x i32> [[F]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <4 x double>, 1, 0) %buffer, i32 %index, i32 0,
      <4 x double> %0)
  ret void
}
