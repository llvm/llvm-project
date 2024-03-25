; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


define void @reg_plus_offset(ptr %a) {
; CHECK:        ldu.global.u32  %r{{[0-9]+}}, [%rd{{[0-9]+}}+32];
; CHECK:        ldu.global.u32  %r{{[0-9]+}}, [%rd{{[0-9]+}}+36];
  %p2 = getelementptr i32, ptr %a, i32 8
  %t1 = call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr %p2, i32 4)
  %p3 = getelementptr i32, ptr %a, i32 9
  %t2 = call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr %p3, i32 4)
  %t3 = mul i32 %t1, %t2
  store i32 %t3, ptr %a
  ret void
}

declare i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr, i32)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
