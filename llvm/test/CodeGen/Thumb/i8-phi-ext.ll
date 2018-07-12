; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m---eabi"

; CHECK-LABEL: test_fn
; CHECK-NOT: uxtb
define dso_local zeroext i8 @test_fn(i32 %x, void (...)* nocapture %f) {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %callee.knr.cast = bitcast void (...)* %f to void ()*
  tail call void %callee.knr.cast() #1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %z.0 = phi i8 [ 3, %if.then ], [ 0, %entry ]
  ret i8 %z.0
}
