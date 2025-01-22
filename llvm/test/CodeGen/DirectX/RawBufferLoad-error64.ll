; We use llc for this test so that we don't abort after the first error.
; RUN: not llc %s -o /dev/null 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.2-compute"

declare void @v4f64_user(<4 x double>)

; Can't load 64 bit types directly until SM6.3 (byteaddressbuf.Load<int64_t4>)
; CHECK: error:
; CHECK-SAME: in function loadv4f64_byte
; CHECK-SAME: Cannot create RawBufferLoad operation: Invalid overload type
define void @loadv4f64_byte(i32 %offset) "hlsl.export" {
  %buffer = call target("dx.RawBuffer", i8, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  %load = call {<4 x double>, i1} @llvm.dx.resource.load.rawbuffer.v4i64(
      target("dx.RawBuffer", i8, 0, 0, 0) %buffer, i32 %offset, i32 0)
  %data = extractvalue {<4 x double>, i1} %load, 0

  call void @v4f64_user(<4 x double> %data)

  ret void
}
