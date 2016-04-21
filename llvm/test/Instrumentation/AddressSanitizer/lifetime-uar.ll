; Test handling of llvm.lifetime intrinsics in UAR mode.
; RUN: opt < %s -asan -asan-module -asan-use-after-return -asan-use-after-scope -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

define i32 @basic_test() sanitize_address {
  ; CHECK-LABEL: define i32 @basic_test()

entry:
  %retval = alloca i32, align 4
  %c = alloca i8, align 1

  call void @llvm.lifetime.start(i64 1, i8* %c)
  ; Memory is unpoisoned at llvm.lifetime.start
  ; CHECK: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 1)

  store volatile i32 0, i32* %retval
  store volatile i8 0, i8* %c, align 1

  call void @llvm.lifetime.end(i64 1, i8* %c)
  ; Memory is poisoned at llvm.lifetime.end
  ; CHECK: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 1)

  ; No need to unpoison memory at function exit in UAR mode.
  ; CHECK-NOT: @__asan_unpoison_stack_memory
  ; CHECK: ret void

  ret i32 0
}

