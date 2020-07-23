; Test handling of llvm.lifetime intrinsics in UAR/UAS modes.
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-after-return=0 -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-use-after-return=0 -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-after-return=1 -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-use-after-return=1 -asan-use-after-scope=0 -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-after-return=0 -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS
; RUN: opt < %s -passes='asan-pipeline' -asan-use-after-return=0 -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-after-return=1 -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS
; RUN: opt < %s -passes='asan-pipeline' -asan-use-after-return=1 -asan-use-after-scope=1 -S | FileCheck %s --check-prefix=CHECK-UAS

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind

define i32 @basic_test() sanitize_address {
  ; CHECK-LABEL: define i32 @basic_test()

entry:
  %retval = alloca i32, align 4
  %c = alloca i8, align 1

  ; Memory is poisoned in prologue: F1F1F1F104F3F8F2
  ; CHECK-UAS: store i64 -866676825215864335, i64* %{{[0-9]+}}

  call void @llvm.lifetime.start.p0i8(i64 1, i8* %c)
  ; Memory is unpoisoned at llvm.lifetime.start: 01
  ; CHECK-UAS: store i8 1, i8* %{{[0-9]+}}

  store volatile i32 0, i32* %retval
  store volatile i8 0, i8* %c, align 1

  call void @llvm.lifetime.end.p0i8(i64 1, i8* %c)
  ; Memory is poisoned at llvm.lifetime.end: F8
  ; CHECK-UAS: store i8 -8, i8* %{{[0-9]+}}

  ; Unpoison memory at function exit in UAS mode.
  ; CHECK-UAS: store i64 0, i64* %{{[0-9]+}}
  ; CHECK-UAS: ret i32 0
  ret i32 0
}

; No poisoning/poisoning at all in plain mode.
; CHECK-NOT: __asan_poison_stack_memory
; CHECK-NOT: __asan_unpoison_stack_memory
