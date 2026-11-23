; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @f32_user(float)

; CHECK: error:
; CHECK-SAME: in function loadrawzero
; CHECK-SAME: Element index of raw buffer must be poison
define void @loadrawzero(i32 %offset) "hlsl.export" {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  %load = call {float, i1}
      @llvm.dx.resource.load.rawbuffer(
          target("dx.RawBuffer", i8, 0, 0, 0) %buffer,
          i32 %offset,
          i32 0)
  %data = extractvalue {float, i1} %load, 0

  call void @f32_user(float %data)

  ret void
}

; CHECK: error:
; CHECK-SAME: in function loadstructundef
; CHECK-SAME: Element index of structured buffer may not be poison
define void @loadstructundef(i32 %index) "hlsl.export" {
  %buffer = call target("dx.RawBuffer", float, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  %load = call {float, i1}
      @llvm.dx.resource.load.rawbuffer(
          target("dx.RawBuffer", float, 0, 0, 0) %buffer,
          i32 %index,
          i32 poison)
  %data = extractvalue {float, i1} %load, 0
  call void @f32_user(float %data)

  ret void
}

; CHECK: error:
; CHECK-SAME: in function storerawzero
; CHECK-SAME: Element index of raw buffer must be poison
define void @storerawzero(i32 %offset, float %data) {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer,
      i32 %offset, i32 0, float %data)

  ret void
}

; CHECK: error:
; CHECK-SAME: in function storestructundef
; CHECK-SAME: Element index of structured buffer may not be poison
define void @storestructundef(i32 %index, float %data) {
  %buffer = call target("dx.RawBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  call void @llvm.dx.resource.store.rawbuffer(
      target("dx.RawBuffer", float, 1, 0, 0) %buffer,
      i32 %index, i32 poison, float %data)

  ret void
}

declare { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0_0t(target("dx.RawBuffer", i8, 0, 0, 0), i32, i32)
declare { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_0_0_0t(target("dx.RawBuffer", float, 0, 0, 0), i32, i32)
declare void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i8_1_0_0t.f32(target("dx.RawBuffer", i8, 1, 0, 0), i32, i32, float)
declare void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_f32_1_0_0t.f32(target("dx.RawBuffer", float, 1, 0, 0), i32, i32, float)
