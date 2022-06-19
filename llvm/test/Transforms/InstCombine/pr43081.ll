; RUN: opt < %s -passes=instcombine -disable-builtin strlen -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i8* @strchr(i8*, i32)

define i8* @pr43081(i8* %a) {
entry:
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8*, i8** %a.addr, align 8
  %call = call i8* @strchr(i8* %0, i32 0)
  ret i8* %call
; CHECK: call i8* @strchr
}
