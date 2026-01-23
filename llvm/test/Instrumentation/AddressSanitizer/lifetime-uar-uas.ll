; Test handling of llvm.lifetime intrinsics in UAR/UAS modes.
; RUN: opt < %s -passes=asan -asan-use-after-return=never -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-use-after-return=runtime -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-use-after-return=always -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-use-after-return=never -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS
; RUN: opt < %s -passes=asan -asan-use-after-return=runtime -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS
; RUN: opt < %s -passes=asan -asan-use-after-return=always -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i32 @basic_test(i64 %i) sanitize_address {
  ; CHECK-LABEL: define i32 @basic_test(

entry:
  %retval = alloca i32, align 4
  %c = alloca [2 x i8], align 1

  ; Memory is poisoned in prologue: F1F1F1F1F8F3F3F3
  ; CHECK-UAS: store i64 -868082052615769615, ptr %{{[0-9]+}}
  ; CHECK-UAS-SS-NOT: store i64

  call void @llvm.lifetime.start.p0(ptr %c)
  ; Memory is unpoisoned at llvm.lifetime.start: 01
  ; CHECK-UAS: store i8 2, ptr %{{[0-9]+}}

  %ci = getelementptr inbounds [2 x i8], ptr %c, i64 0, i64 %i
  store volatile i32 0, ptr %retval
  store volatile i8 0, ptr %ci, align 1

  call void @llvm.lifetime.end.p0(ptr %c)
  ; Memory is poisoned at llvm.lifetime.end: F8
  ; CHECK-UAS: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-UAS-SS-NOT: store i8 -8,

  ; Unpoison memory at function exit in UAS mode.
  ; CHECK-UAS: store i64 0, ptr %{{[0-9]+}}
  ; CHECK-UAS: ret i32 0
  ret i32 0
}

; No poisoning/poisoning at all in plain mode.
; CHECK-NOT: __asan_poison_stack_memory
; CHECK-NOT: __asan_unpoison_stack_memory
