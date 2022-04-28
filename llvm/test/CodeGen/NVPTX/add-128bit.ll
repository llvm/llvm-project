; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"



define void @foo(i64 %a, i64 %add, i128* %retptr) {
; CHECK:        add.s64
; CHECK:        setp.lt.u64
; CHECK:        setp.lt.u64
; CHECK:        selp.u64
; CHECK:        selp.b64
; CHECK:        add.s64
  %t1 = sext i64 %a to i128
  %add2 = zext i64 %add to i128
  %val = add i128 %t1, %add2
  store i128 %val, i128* %retptr
  ret void
}
