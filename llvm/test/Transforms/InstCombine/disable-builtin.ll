; Test that -disable-builtin works correctly.
;
; RUN: opt < %s -passes=instcombine -disable-builtin strcat -S | FileCheck %s
;
; RUN: not opt < %s -passes=instcombine -disable-builtin foobar -S 2>&1 | FileCheck --check-prefix=FOOBAR %s
; FOOBAR: cannot disable nonexistent builtin function foobar

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@empty = constant [1 x i8] c"\00", align 1

declare ptr @strcat(ptr, ptr)

define ptr @test_strcat(ptr %x) {
; CHECK-LABEL: @test_strcat(
  %ret = call ptr @strcat(ptr %x, ptr @empty)
  ret ptr %ret
; CHECK: call ptr @strcat
}

