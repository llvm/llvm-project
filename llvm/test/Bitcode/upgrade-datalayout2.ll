; Test to make sure datalayout is not automatically upgraded if it does not
; match a possible x86 datalayout.
;
; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s
;
; XFAIL: *
; No implementation of the data layout upgrade ever checked whether the data
; layout was a possible x86 data layout, so the logic that this test aims to
; check was never implemented. We always upgraded data layouts that were not
; possible x86 data layouts, we merely did not previously upgrade this one.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

