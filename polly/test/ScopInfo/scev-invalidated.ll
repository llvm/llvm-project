; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Region: %if.then6---%return
;
target datalayout ="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @arc_either() {
entry:
  %ang2.2.reg2mem = alloca i64
  br i1 undef, label %return, label %if.then6

if.then6:
  %rem7 = srem i64 undef, 1474560
  br i1 false, label %if.else, label %return

if.else:
  %add16 = add nsw i64 %rem7, 1474560
  %rem7.add16 = select i1 undef, i64 %rem7, i64 %add16
  store i64 %rem7.add16, i64* %ang2.2.reg2mem
  br label %return

return:
  ret void
}
