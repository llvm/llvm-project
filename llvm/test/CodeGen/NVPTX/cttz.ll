; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i16 @llvm.cttz.i16(i16, i1) readnone
declare i32 @llvm.cttz.i32(i32, i1) readnone
declare i64 @llvm.cttz.i64(i64, i1) readnone

define i32 @myctpop(i32 %a) {
; CHECK: popc.b32
  %val = call i32 @llvm.cttz.i32(i32 %a, i1 false) readnone
  ret i32 %val
}

define i16 @myctpop16(i16 %a) {
; CHECK: popc.b32
  %val = call i16 @llvm.cttz.i16(i16 %a, i1 false) readnone
  ret i16 %val
}

define i64 @myctpop64(i64 %a) {
; CHECK: popc.b64
  %val = call i64 @llvm.cttz.i64(i64 %a, i1 false) readnone
  ret i64 %val
}


define i32 @myctpop_2(i32 %a) {
; CHECK: popc.b32
  %val = call i32 @llvm.cttz.i32(i32 %a, i1 true) readnone
  ret i32 %val
}

define i16 @myctpop16_2(i16 %a) {
; CHECK: popc.b32
  %val = call i16 @llvm.cttz.i16(i16 %a, i1 true) readnone
  ret i16 %val
}

define i64 @myctpop64_2(i64 %a) {
; CHECK: popc.b64
  %val = call i64 @llvm.cttz.i64(i64 %a, i1 true) readnone
  ret i64 %val
}

; The cttz is expanded to popc.b64 during legalization, so the compare is
; combined with the 32-bit count of a 64-bit value.
define i1 @mycttz64_ult_2(i64 %a) {
; CHECK-LABEL: mycttz64_ult_2(
; CHECK: popc.b64 [[CNT:%r[0-9]+]],
; CHECK: setp.lt.u32 {{%p[0-9]+}}, [[CNT]], 2;
  %val = call i64 @llvm.cttz.i64(i64 %a, i1 true) readnone
  %trunc = trunc i64 %val to i32
  %cmp = icmp ult i32 %trunc, 2
  ret i1 %cmp
}
