; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Sub-byte and non-power-of-two integers must be widened to a legal PTX type.

; CHECK: .visible .global .align 1 .u8 g4;
; CHECK: .visible .global .align 4 .u32 g24;
@g4 = addrspace(1) global i4 0
@g24 = addrspace(1) global i24 0

; CHECK: .entry sub_byte
; CHECK:   .param .u8 sub_byte_param_0
; CHECK:   .param .u64 .ptr .align 1 sub_byte_param_1
; CHECK-DAG: ld.param.b8 {{%r[0-9]+}}, [sub_byte_param_0];
define ptx_kernel void @sub_byte(i4 %p, ptr %out) {
  %val = zext i4 %p to i32
  store i32 %val, ptr %out
  ret void
}

; CHECK: .entry not_byte_multiple
; CHECK:   .param .u16 not_byte_multiple_param_0
; CHECK-DAG: ld.param.b16 {{%r[0-9]+}}, [not_byte_multiple_param_0];
define ptx_kernel void @not_byte_multiple(i12 %p, ptr %out) {
  %val = zext i12 %p to i32
  store i32 %val, ptr %out
  ret void
}

; CHECK: .entry not_power_of_two
; CHECK:   .param .u32 not_power_of_two_param_0
; CHECK-DAG: ld.param.b32 {{%r[0-9]+}}, [not_power_of_two_param_0];
define ptx_kernel void @not_power_of_two(i24 %p, ptr %out) {
  %val = zext i24 %p to i32
  store i32 %val, ptr %out
  ret void
}

; CHECK: .entry wide
; CHECK:   .param .u64 wide_param_0
; CHECK-DAG: ld.param.b64 {{%rd[0-9]+}}, [wide_param_0];
define ptx_kernel void @wide(i48 %p, ptr %out) {
  %val = zext i48 %p to i64
  store i64 %val, ptr %out
  ret void
}
