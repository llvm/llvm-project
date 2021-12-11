; Test for -msan-disable-checks, which should treat every function in the file
; as if it didn't have the sanitize_memory attribute.
; RUN: opt < %s -msan-check-access-address=0 -S -passes='module(msan-module),function(msan)' 2>&1 | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CHECK,INSTR %s
; RUN: opt < %s -msan-check-access-address=0 -S -passes='module(msan-module),function(msan)' -msan-disable-checks=1 2>&1 | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CHECK,NOSANITIZE %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar()

define i32 @SanitizeFn(i32 %x) uwtable sanitize_memory {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret i32 %x
}

; CHECK-LABEL: @SanitizeFn
; INSTR: @__msan_warning
; NOSANITIZE-NOT: @__msan_warning
; NOSANITIZE: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


define i32 @NoSanitizeFn(i32 %x) uwtable {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret i32 %x
}


; CHECK-LABEL: @NoSanitizeFn
; CHECK-NOT: @__msan_warning
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32

