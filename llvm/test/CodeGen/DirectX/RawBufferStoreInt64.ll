; RUN: opt -mtriple=dxil-pc-shadermodel6.2-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK62
; RUN: opt -mtriple=dxil-pc-shadermodel6.3-compute -S -dxil-intrinsic-expansion %s | FileCheck %s --check-prefixes=CHECK,CHECK63

define void @storei64(i64 %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", i64, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.RawBuffer", i64, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i64_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: call void @llvm.dx.resource.store.rawbuffer
  ; CHECK63-SAME: target("dx.RawBuffer", i64, 1, 0) [[Buf]], i32 %index, i32 0, i64 %0) 

  ; check we split the i64 and store the lo and hi bits
  ; CHECK62: [[A:%.*]] = trunc i64 %0 to i32
  ; CHECK62: [[B:%.*]] = lshr i64 %0, 32
  ; CHECK62: [[C:%.*]] = trunc i64 [[B]] to i32
  ; CHECK62: [[Vec1:%.*]] = insertelement <2 x i32> poison, i32 [[A]], i32 0
  ; CHECK62: [[Vec2:%.*]] = insertelement <2 x i32> [[Vec1]], i32 [[C]], i32 1
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i64_1_0t.v2i32(
  ; CHECK62-SAME: target("dx.RawBuffer", i64, 1, 0) [[Buf]], i32 %index, i32 0, <2 x i32> [[Vec2]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", i64, 1, 0) %buffer, i32 %index, i32 0,
      i64 %0)
  ret void
}

define void @storev2i64(<2 x i64> %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", <2 x i64>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v2i64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.RawBuffer", <2 x i64>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v2i64_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: call void @llvm.dx.resource.store.rawbuffer
  ; CHECK63-SAME: target("dx.RawBuffer", <2 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <2 x i64> %0)

  ; CHECK62: [[A:%.*]] = trunc <2 x i64> %0 to <2 x i32>
  ; CHECK62: [[B:%.*]] = lshr <2 x i64> %0, splat (i64 32)
  ; CHECK62: [[C:%.*]] = trunc <2 x i64> [[B]] to <2 x i32>
  ; CHECK62: [[Vec:%.*]] = shufflevector <2 x i32> [[A]], <2 x i32> [[C]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v2i64_1_0t.v4i32(
  ; CHECK62-SAME: target("dx.RawBuffer", <2 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i32> [[Vec]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <2 x i64>, 1, 0) %buffer, i32 %index, i32 0,
      <2 x i64> %0)
  ret void
}

define void @storev3i64(<3 x i64> %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", <3 x i64>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v3i64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.RawBuffer", <3 x i64>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v3i64_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: call void @llvm.dx.resource.store.rawbuffer
  ; CHECK63-SAME: target("dx.RawBuffer", <3 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <3 x i64> %0)

  ; CHECK62: [[A:%.*]] = trunc <3 x i64> %0 to <3 x i32>
  ; CHECK62: [[B:%.*]] = lshr <3 x i64> %0, splat (i64 32)
  ; CHECK62: [[C:%.*]] = trunc <3 x i64> [[B]] to <3 x i32>
  ; CHECK62: [[D:%.*]] = shufflevector <3 x i32> [[A]], <3 x i32> [[C]], <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
  ; CHECK62: [[E:%.*]] = shufflevector <6 x i32> [[D]], <6 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v3i64_1_0t.v4i32(target("dx.RawBuffer", <3 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i32> [[E]])
  ; CHECK62: [[F:%.*]] = shufflevector <6 x i32> [[D]], <6 x i32> poison, <2 x i32> <i32 4, i32 5>
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v3i64_1_0t.v2i32(target("dx.RawBuffer", <3 x i64>, 1, 0) [[Buf]], i32 %index, i32 16, <2 x i32> [[F]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <3 x i64>, 1, 0) %buffer, i32 %index, i32 0,
      <3 x i64> %0)
  ret void
}

define void @storev4i64(<4 x i64> %0, i32 %index) {
  ; CHECK: [[Buf:%.*]] = tail call target("dx.RawBuffer", <4 x i64>, 1, 0)
  ; CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4i64_1_0t(
  ; CHECK-SAME: i32 0, i32 0, i32 1, i32 0, ptr null)
  %buffer = tail call target("dx.RawBuffer", <4 x i64>, 1, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4i64_1_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  ; check we don't modify the code in sm6.3 or later
  ; CHECK63: call void @llvm.dx.resource.store.rawbuffer
  ; CHECK63-SAME: target("dx.RawBuffer", <4 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i64> %0)

  ; CHECK62: [[A:%.*]] = trunc <4 x i64> %0 to <4 x i32>
  ; CHECK62: [[B:%.*]] = lshr <4 x i64> %0, splat (i64 32)
  ; CHECK62: [[C:%.*]] = trunc <4 x i64> [[B]] to <4 x i32>
  ; CHECK62: [[D:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[C]], <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  ; CHECK62: [[E:%.*]] = shufflevector <8 x i32> [[D]], <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4i64_1_0t.v4i32(target("dx.RawBuffer", <4 x i64>, 1, 0) [[Buf]], i32 %index, i32 0, <4 x i32> [[E]])
  ; CHECK62: [[F:%.*]] = shufflevector <8 x i32> [[D]], <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ; CHECK62: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4i64_1_0t.v4i32(target("dx.RawBuffer", <4 x i64>, 1, 0) [[Buf]], i32 %index, i32 16, <4 x i32> [[F]])
  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", <4 x i64>, 1, 0) %buffer, i32 %index, i32 0,
      <4 x i64> %0)
  ret void
}
