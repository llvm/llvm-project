; RUN: opt -S -dxil-forward-handle-accesses -mtriple=dxil--shadermodel6.3-library %s | FileCheck %s

%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", <4 x float>, 1, 0) }

@_ZL2In = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
@_ZL3Out = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4

define void @main() #1 {
entry:
  %this.addr.i.i.i = alloca ptr, align 4
  %this.addr.i.i = alloca ptr, align 4
  %this.addr.i1 = alloca ptr, align 4
  %Index.addr.i2 = alloca i32, align 4
  %this.addr.i = alloca ptr, align 4
  %Index.addr.i = alloca i32, align 4
  ; CHECK: [[IN:%.*]] = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  %_ZL2In_h.i.i = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %_ZL2In_h.i.i, ptr @_ZL2In, align 4
  store ptr @_ZL2In, ptr %this.addr.i.i, align 4
  %this1.i.i = load ptr, ptr %this.addr.i.i, align 4
  ; CHECK: [[OUT:%.*]] = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_v4f32_1_0t(i32 100, i32 0, i32 1, i32 0, ptr null)
  %_ZL3Out_h.i.i = call target("dx.RawBuffer", <4 x float>, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_v4f32_1_0t(i32 100, i32 0, i32 1, i32 0, ptr null)
  store target("dx.RawBuffer", <4 x float>, 1, 0) %_ZL3Out_h.i.i, ptr @_ZL3Out, align 4
  store ptr @_ZL3Out, ptr %this.addr.i.i.i, align 4
  %this1.i.i.i = load ptr, ptr %this.addr.i.i.i, align 4
  store ptr @_ZL2In, ptr %this.addr.i1, align 4
  store i32 0, ptr %Index.addr.i2, align 4
  %this1.i3 = load ptr, ptr %this.addr.i1, align 4
  ; CHECK-NOT: load target("dx.RawBuffer", <4 x float>, 1, 0)
  %0 = load target("dx.RawBuffer", <4 x float>, 1, 0), ptr %this1.i3, align 4
  %1 = load i32, ptr %Index.addr.i2, align 4
  ; CHECK: call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_v4f32_1_0t(target("dx.RawBuffer", <4 x float>, 1, 0) [[IN]],
  %2 = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_v4f32_1_0t(target("dx.RawBuffer", <4 x float>, 1, 0) %0, i32 %1, i32 0)
  %3 = extractvalue { <4 x float>, i1 } %2, 0
  store ptr @_ZL3Out, ptr %this.addr.i, align 4
  store i32 0, ptr %Index.addr.i, align 4
  %this1.i = load ptr, ptr %this.addr.i, align 4
  ; CHECK-NOT: load target("dx.RawBuffer", <4 x float>, 1, 0)
  %4 = load target("dx.RawBuffer", <4 x float>, 1, 0), ptr %this1.i, align 4
  %5 = load i32, ptr %Index.addr.i, align 4
  ; CHECK: call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f32_1_0t.v4f32(target("dx.RawBuffer", <4 x float>, 1, 0) [[OUT]],
  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4f32_1_0t.v4f32(target("dx.RawBuffer", <4 x float>, 1, 0) %4, i32 %5, i32 0, <4 x float> %3)
  ret void
}
