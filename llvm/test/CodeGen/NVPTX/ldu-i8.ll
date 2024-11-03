; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr, i32)

define i8 @foo(ptr %a) {
; Ensure we properly truncate off the high-order 24 bits
; CHECK:        ldu.global.u8
; CHECK:        cvt.u32.u16
; CHECK:        and.b32         %r{{[0-9]+}}, %r{{[0-9]+}}, 255
  %val = tail call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr %a, i32 4)
  ret i8 %val
}
