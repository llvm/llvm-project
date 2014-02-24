; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Check that vastart handling doesn't get between testb and je for the branch.
define i32 @check_flag(i32 %flags, ...) nounwind {
entry:
; CHECK: {{^}} testb $2, %bh
; CHECK-NOT: test
; CHECK: {{^}} je
  %and = and i32 %flags, 512
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %hasflag = phi i32 [ 1, %if.then ], [ 0, %entry ]
  ret i32 %hasflag
}

