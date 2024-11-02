; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}


;; Integer conversions happen inplicitly by loading/storing the proper types


; i16

define i16 @cvt_i16_i32(i32 %x) {
; CHECK: ld.param.u16 %r[[R0:[0-9]+]], [cvt_i16_i32_param_{{[0-9]+}}]
; CHECK: st.param.b32 [func_retval{{[0-9]+}}+0], %r[[R0]]
; CHECK: ret
  %a = trunc i32 %x to i16
  ret i16 %a
}

define i16 @cvt_i16_i64(i64 %x) {
; CHECK: ld.param.u16 %r[[R0:[0-9]+]], [cvt_i16_i64_param_{{[0-9]+}}]
; CHECK: st.param.b32 [func_retval{{[0-9]+}}+0], %r[[R0]]
; CHECK: ret
  %a = trunc i64 %x to i16
  ret i16 %a
}



; i32

define i32 @cvt_i32_i16(i16 %x) {
; CHECK: ld.param.u16 %r[[R0:[0-9]+]], [cvt_i32_i16_param_{{[0-9]+}}]
; CHECK: st.param.b32 [func_retval{{[0-9]+}}+0], %r[[R0]]
; CHECK: ret
  %a = zext i16 %x to i32
  ret i32 %a
}

define i32 @cvt_i32_i64(i64 %x) {
; CHECK: ld.param.u32 %r[[R0:[0-9]+]], [cvt_i32_i64_param_{{[0-9]+}}]
; CHECK: st.param.b32 [func_retval{{[0-9]+}}+0], %r[[R0]]
; CHECK: ret
  %a = trunc i64 %x to i32
  ret i32 %a
}



; i64

define i64 @cvt_i64_i16(i16 %x) {
; CHECK: ld.param.u16 %rd[[R0:[0-9]+]], [cvt_i64_i16_param_{{[0-9]+}}]
; CHECK: st.param.b64 [func_retval{{[0-9]+}}+0], %rd[[R0]]
; CHECK: ret
  %a = zext i16 %x to i64
  ret i64 %a
}

define i64 @cvt_i64_i32(i32 %x) {
; CHECK: ld.param.u32 %rd[[R0:[0-9]+]], [cvt_i64_i32_param_{{[0-9]+}}]
; CHECK: st.param.b64 [func_retval{{[0-9]+}}+0], %rd[[R0]]
; CHECK: ret
  %a = zext i32 %x to i64
  ret i64 %a
}
