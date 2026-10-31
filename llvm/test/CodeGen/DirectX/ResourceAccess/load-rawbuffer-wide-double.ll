; RUN: opt -S -dxil-resource-access -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.2-compute %s | FileCheck %s

define <8 x double> @loadv8f64_byte(i32 %index) {
  %buffer = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 0, 0) %buffer, i32 %index)

  ; CHECK: call { <4 x i32>, i1 } [[LOAD_INTRINSIC:@llvm\.dx\.resource\.load\.rawbuffer[^(]*]]([[BUFFER_TY:target\("dx\.RawBuffer", i8, 0, 0\)]] %buffer, i32 %index, i32 poison)
  ; CHECK: %[[INDEX16:.*]] = add i32 %index, 16
  ; CHECK: call { <4 x i32>, i1 } [[LOAD_INTRINSIC]]([[BUFFER_TY]] %buffer, i32 %[[INDEX16]], i32 poison)
  ; CHECK: %[[INDEX32:.*]] = add i32 %index, 32
  ; CHECK: call { <4 x i32>, i1 } [[LOAD_INTRINSIC]]([[BUFFER_TY]] %buffer, i32 %[[INDEX32]], i32 poison)
  ; CHECK: %[[INDEX48:.*]] = add i32 %[[INDEX32]], 16
  ; CHECK: call { <4 x i32>, i1 } [[LOAD_INTRINSIC]]([[BUFFER_TY]] %buffer, i32 %[[INDEX48]], i32 poison)
  %data = load <8 x double>, ptr %ptr

  ret <8 x double> %data
}
