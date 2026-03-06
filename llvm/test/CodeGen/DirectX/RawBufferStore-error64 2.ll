; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.2-compute"

; Can't store 64 bit types directly until SM6.3 (byteaddressbuf.Store<int64_t4>)
; CHECK: error:
; CHECK-SAME: in function storev4f64_byte
; CHECK-SAME: Cannot create RawBufferStore operation: Invalid overload type
define void @storev4f64_byte(i32 %offset, <4 x double> %data) "hlsl.export" {
  %buffer = call target("dx.RawBuffer", i8, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.resource.store.rawbuffer.v4i64(
      target("dx.RawBuffer", i8, 1, 0, 0) %buffer,
      i32 %offset, i32 0, <4 x double> %data)

  ret void
}
