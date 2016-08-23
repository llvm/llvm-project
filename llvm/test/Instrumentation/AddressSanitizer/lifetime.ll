; Test handling of llvm.lifetime intrinsics.
; RUN: opt < %s -asan -asan-module -asan-use-after-scope -asan-use-after-return=0 -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -asan-use-after-scope -asan-use-after-return=0 -asan-instrument-dynamic-allocas=0 -S | FileCheck %s --check-prefix=CHECK-NO-DYNAMIC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

define void @lifetime_no_size() sanitize_address {
entry:
  %i = alloca i32, align 4
  %i.ptr = bitcast i32* %i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %i.ptr)
  store volatile i8 0, i8* %i.ptr
  call void @llvm.lifetime.end(i64 -1, i8* %i.ptr)

; Check that lifetime with no size are ignored.
; CHECK-LABEL: define void @lifetime_no_size()
; CHECK-NOT: @__asan_poison_stack_memory
; CHECK-NOT: @__asan_unpoison_stack_memory
; CHECK: ret void
  ret void
}

; Generic case of lifetime analysis.
define void @lifetime() sanitize_address {
  ; CHECK-LABEL: define void @lifetime()

  ; Regular variable lifetime intrinsics.
  %i = alloca i32, align 4
  %i.ptr = bitcast i32* %i to i8*
  call void @llvm.lifetime.start(i64 3, i8* %i.ptr)
  store volatile i8 0, i8* %i.ptr
  ; Memory is unpoisoned at llvm.lifetime.start
  ; CHECK: %[[VAR:[^ ]*]] = ptrtoint i32* %{{[^ ]+}} to i64
  ; CHECK-NEXT: call void @__asan_unpoison_stack_memory(i64 %[[VAR]], i64 3)
  call void @llvm.lifetime.end(i64 4, i8* %i.ptr)
  call void @llvm.lifetime.end(i64 2, i8* %i.ptr)
  ; Memory is poisoned at every call to llvm.lifetime.end
  ; CHECK: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 4)
  ; CHECK: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 2)

  ; Lifetime intrinsics for array.
  %arr = alloca [10 x i32], align 16
  %arr.ptr = bitcast [10 x i32]* %arr to i8*
  call void @llvm.lifetime.start(i64 40, i8* %arr.ptr)
  store volatile i8 0, i8* %arr.ptr
  ; CHECK: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 40)
  ; CHECK-NO-DYNAMIC-NOT: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 40)
  call void @llvm.lifetime.end(i64 40, i8* %arr.ptr)
  ; CHECK: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 40)
  ; CHECK-NO-DYNAMIC-NOT: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 40)

  ; One more lifetime start/end for the same variable %i.
  call void @llvm.lifetime.start(i64 4, i8* %i.ptr)
  store volatile i8 0, i8* %i.ptr
  ; CHECK: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 4)
  call void @llvm.lifetime.end(i64 4, i8* %i.ptr)
  ; CHECK: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 4)

  ; Memory is unpoisoned at function exit (only once).
  ; CHECK: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 {{.*}})
  ; CHECK-NOT: @__asan_unpoison_stack_memory
  ; CHECK: ret void
  ret void
}

; Check that arguments of lifetime may come from phi nodes.
define void @phi_args(i1 %x) sanitize_address {
  ; CHECK-LABEL: define void @phi_args(i1 %x)

entry:
  %i = alloca i64, align 4
  %i.ptr = bitcast i64* %i to i8*
  call void @llvm.lifetime.start(i64 8, i8* %i.ptr)
  store volatile i8 0, i8* %i.ptr
  ; CHECK: __asan_unpoison_stack_memory
  br i1 %x, label %bb0, label %bb1

bb0:
  %i.ptr2 = bitcast i64* %i to i8*
  br label %bb1

bb1:
  %i.phi = phi i8* [ %i.ptr, %entry ], [ %i.ptr2, %bb0 ]
  call void @llvm.lifetime.end(i64 8, i8* %i.phi)
  ; CHECK: __asan_poison_stack_memory
  ; CHECK: ret void
  ret void
}

; Check that arguments of lifetime may come from getelementptr nodes.
define void @getelementptr_args() sanitize_address{
  ; CHECK-LABEL: define void @getelementptr_args
entry:
  %x = alloca [1024 x i8], align 16
  %d = alloca i8*, align 8

  %0 = getelementptr inbounds [1024 x i8], [1024 x i8]* %x, i64 0, i64 0
  call void @llvm.lifetime.start(i64 1024, i8* %0)
  ; CHECK: __asan_unpoison_stack_memory

  store i8* %0, i8** %d, align 8

  call void @llvm.lifetime.end(i64 1024, i8* %0)
  ; CHECK: __asan_poison_stack_memory

  ret void
  ; CHECK: store i32 0
  ; CHECK: store i64 0
  ; CHECK: store i64 0
  ; CHECK-NEXT: __asan_unpoison_stack_memory
}

define void @zero_sized(i64 %a) #0 {
; CHECK-LABEL: define void @zero_sized(i64 %a)

entry:
  %a.addr = alloca i64, align 8
  %b = alloca [0 x i8], align 1
  store i64 %a, i64* %a.addr, align 8
  %0 = bitcast [0 x i8]* %b to i8*
  call void @llvm.lifetime.start(i64 0, i8* %0) #2
  ; CHECK-NOT: call void @__asan_unpoison_stack_memory
  %1 = bitcast [0 x i8]* %b to i8*
  call void @llvm.lifetime.end(i64 0, i8* %1) #2
  ; CHECK-NOT: call void @__asan_poison_stack_memory

  ret void
}
