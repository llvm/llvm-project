; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @myctpop(i32 %a) {
; CHECK: popc.b32
  %val = tail call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %val
}

define i16 @myctpop16(i16 %a) {
; CHECK: popc.b32
  %val = tail call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %val
}

define i64 @myctpop64(i64 %a) {
; CHECK: popc.b64
  %val = tail call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %val
}

declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
