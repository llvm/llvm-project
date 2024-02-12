; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: @bswap16
; CHECK:   ld.param.u16    %rs1, [bswap16_param_0];
; CHECK:   shr.u16         %rs2, %rs1, 8;
; CHECK:   shl.b16         %rs3, %rs1, 8;
; CHECK:   or.b16          %rs4, %rs3, %rs2;
; CHECK:   cvt.u32.u16     %r1, %rs4;
; CHECK:   st.param.b32    [func_retval0+0], %r1;

define i16 @bswap16(i16 %a) {
  %b = tail call i16 @llvm.bswap.i16(i16 %a)
  ret i16 %b
}

; CHECK-LABEL: @bswap32
; CHECK:    ld.param.u32    %r1, [bswap32_param_0];
; CHECK:    prmt.b32        %r2, %r1, 0, 291;
; CHECK:    st.param.b32    [func_retval0+0], %r2;

define i32 @bswap32(i32 %a) {
  %b = tail call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %b
}

; CHECK-LABEL: @bswapv2i16
; CHECK:    ld.param.u32    %r1, [bswapv2i16_param_0];
; CHECK:    prmt.b32        %r2, %r1, 0, 8961;
; CHECK:    st.param.b32    [func_retval0+0], %r2;

define <2 x i16> @bswapv2i16(<2 x i16> %a) #0 {
  %b = tail call <2 x i16> @llvm.bswap.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

; CHECK-LABEL: @bswap64
; CHECK:    ld.param.u64    %rd1, [bswap64_param_0];
; CHECK:    { .reg .b32 tmp; mov.b64 {%r1, tmp}, %rd1; }
; CHECK:    prmt.b32        %r2, %r1, 0, 291;
; CHECK:    { .reg .b32 tmp; mov.b64 {tmp, %r3}, %rd1; }
; CHECK:    prmt.b32        %r4, %r3, 0, 291;
; CHECK:    mov.b64         %rd2, {%r4, %r2};
; CHECK:    st.param.b64    [func_retval0+0], %rd2;

define i64 @bswap64(i64 %a) {
  %b = tail call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %b
}

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare <2 x i16> @llvm.bswap.v2i16(<2 x i16>)
declare i64 @llvm.bswap.i64(i64)
