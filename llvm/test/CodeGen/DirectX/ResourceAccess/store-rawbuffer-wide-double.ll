; RUN: opt -S -dxil-resource-access -dxil-intrinsic-expansion \
; RUN:   -mtriple=dxil-pc-shadermodel6.2-compute %s | FileCheck %s

define void @storev8f64_byte(i32 %index, <8 x double> %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.RawBuffer", i8, 1, 0) %buffer, i32 %index)

  ; CHECK: call void [[STORE:@llvm\.dx\.resource\.store\.rawbuffer[^(]*]]([[BUFFER_TY:target\("dx\.RawBuffer", i8, 1, 0\)]] %buffer, i32 %index, [[STORE_TAIL:i32 poison, <4 x i32>]]
  ; CHECK: %[[INDEX16:.*]] = add i32 %index, 16
  ; CHECK: call void [[STORE]]([[BUFFER_TY]] %buffer, i32 %[[INDEX16]], [[STORE_TAIL]]
  ; CHECK: %[[INDEX32:.*]] = add i32 %index, 32
  ; CHECK: call void [[STORE]]([[BUFFER_TY]] %buffer, i32 %[[INDEX32]], [[STORE_TAIL]]
  ; CHECK: %[[INDEX48:.*]] = add i32 %[[INDEX32]], 16
  ; CHECK: call void [[STORE]]([[BUFFER_TY]] %buffer, i32 %[[INDEX48]], [[STORE_TAIL]]
  store <8 x double> %data, ptr %ptr

  ret void
}
