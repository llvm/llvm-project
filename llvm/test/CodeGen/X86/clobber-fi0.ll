; RUN: llc < %s -verify-machineinstrs -mcpu=generic -mtriple=x86_64-linux | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

; In the code below we need to copy the EFLAGS because of scheduling constraints.
; When copying the EFLAGS we need to write to the stack with push/pop. This forces
; us to emit the prolog.

; CHECK: main
; CHECK: subq{{.*}}rsp
; CHECK: ret
define i32 @main(i32 %arg, i8** %arg1) nounwind {
bb:
  %tmp = alloca i32, align 4                      ; [#uses=3 type=i32*]
  %tmp2 = alloca i32, align 4                     ; [#uses=3 type=i32*]
  %tmp3 = alloca i32                              ; [#uses=1 type=i32*]
  store volatile i32 1, i32* %tmp, align 4
  store volatile i32 1, i32* %tmp2, align 4
  br label %bb4

bb4:                                              ; preds = %bb4, %bb
  %tmp6 = load volatile i32, i32* %tmp2, align 4                ; [#uses=1 type=i32]
  %tmp7 = add i32 %tmp6, -1                       ; [#uses=2 type=i32]
  store volatile i32 %tmp7, i32* %tmp2, align 4
  %tmp8 = icmp eq i32 %tmp7, 0                    ; [#uses=1 type=i1]
  %tmp9 = load volatile i32, i32* %tmp                          ; [#uses=1 type=i32]
  %tmp10 = add i32 %tmp9, -1              ; [#uses=1 type=i32]
  store volatile i32 %tmp10, i32* %tmp3
  br i1 %tmp8, label %bb11, label %bb4

bb11:                                             ; preds = %bb4
  %tmp12 = load volatile i32, i32* %tmp, align 4                ; [#uses=1 type=i32]
  ret i32 %tmp12
}


