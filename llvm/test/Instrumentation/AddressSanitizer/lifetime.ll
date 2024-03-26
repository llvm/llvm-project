; Test handling of llvm.lifetime intrinsics.
; RUN: opt < %s -passes=asan -asan-use-after-scope -asan-use-after-return=never -asan-use-stack-safety=0 -S | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
; RUN: opt < %s -passes=asan -asan-use-after-scope -asan-use-after-return=never -asan-use-stack-safety=0 -asan-instrument-dynamic-allocas=0 -S | FileCheck %s --check-prefixes=CHECK,CHECK-NO-DYNAMIC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

; CHECK-LABEL: define void @lifetime_no_size(
define void @lifetime_no_size(i64 %i) sanitize_address {
entry:
  %a = alloca [2 x i32], align 4

  ; Poison memory in prologue: 0xf3f3f300f1f1f1f1
  ; CHECK: store i64 -868083117767659023, ptr %[[#]]

  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  ; Check that lifetime with no size are ignored.
  ; CHECK-NOT: store
  ; CHECK: call void @llvm.lifetime.start

  %ai = getelementptr inbounds [2 x i32], ptr %a, i64 0, i64 %i
  store volatile i8 0, ptr %ai, align 4
  ; CHECK: store volatile

  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  ; Check that lifetime with no size are ignored.
  ; CHECK-NOT: store
  ; CHECK: call void @llvm.lifetime.end

  ; Unpoison stack frame on exit.
  ; CHECK: store i64 0, ptr %{{[0-9]+}}
  ; CHECK: ret void
  ret void
}

; Generic case of lifetime analysis.
define void @lifetime() sanitize_address {
  ; CHECK-LABEL: define void @lifetime()

  ; Regular variable lifetime intrinsics.
  %i = alloca i32, align 4

  ; Poison memory in prologue: F1F1F1F1F8F3F3F3
  ; CHECK: store i64 -868082052615769615, ptr %{{[0-9]+}}

  ; Memory is unpoisoned at llvm.lifetime.start
  call void @llvm.lifetime.start.p0(i64 3, ptr %i)
  ; CHECK: store i8 4, ptr %{{[0-9]+}}
  ; CHECK-NEXT: llvm.lifetime.start

  store volatile i8 0, ptr %i
  ; CHECK: store volatile

  call void @llvm.lifetime.end.p0(i64 4, ptr %i)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: call void @llvm.lifetime.end

  ; Memory is poisoned at every call to llvm.lifetime.end
  call void @llvm.lifetime.end.p0(i64 2, ptr %i)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: call void @llvm.lifetime.end

  ; Lifetime intrinsics for array.
  %arr = alloca [10 x i32], align 16

  call void @llvm.lifetime.start.p0(i64 40, ptr %arr)
  ; CHECK-DEFAULT: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 40)
  ; CHECK-NO-DYNAMIC-NOT: call void @__asan_unpoison_stack_memory(i64 %{{[^ ]+}}, i64 40)

  store volatile i8 0, ptr %arr
  ; CHECK: store volatile

  call void @llvm.lifetime.end.p0(i64 40, ptr %arr)
  ; CHECK-DEFAULT: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 40)
  ; CHECK-NO-DYNAMIC-NOT: call void @__asan_poison_stack_memory(i64 %{{[^ ]+}}, i64 40)

  ; One more lifetime start/end for the same variable %i.
  call void @llvm.lifetime.start.p0(i64 2, ptr %i)
  ; CHECK: store i8 4, ptr %{{[0-9]+}}
  ; CHECK-NEXT: llvm.lifetime.start

  store volatile i8 0, ptr %i
  ; CHECK: store volatile

  call void @llvm.lifetime.end.p0(i64 4, ptr %i)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: llvm.lifetime.end

  ; Memory is unpoisoned at function exit (only once).
  ; CHECK: store i64 0, ptr %{{[0-9]+}}
  ; CHECK-NEXT: ret void
  ret void
}

; Check that arguments of lifetime may come from phi nodes.
define void @phi_args(i1 %x) sanitize_address {
  ; CHECK-LABEL: define void @phi_args(i1 %x)

entry:
  %i = alloca i64, align 4

  ; Poison memory in prologue: F1F1F1F1F8F3F3F3
  ; CHECK: store i64 -868082052615769615, ptr %{{[0-9]+}}

  call void @llvm.lifetime.start.p0(i64 8, ptr %i)
  ; CHECK: store i8 0, ptr %{{[0-9]+}}
  ; CHECK-NEXT: llvm.lifetime.start

  store volatile i8 0, ptr %i
  ; CHECK: store volatile

  br i1 %x, label %bb0, label %bb1

bb0:
  br label %bb1

bb1:
  %i.phi = phi ptr [ %i, %entry ], [ %i, %bb0 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr %i.phi)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: llvm.lifetime.end

  ret void
  ; CHECK: store i64 0, ptr %{{[0-9]+}}
  ; CHECK-NEXT: ret void
}

; Check that arguments of lifetime may come from getelementptr nodes.
define void @getelementptr_args(i64 %i) sanitize_address{
  ; CHECK-LABEL: define void @getelementptr_args
entry:
  %x = alloca [1024 x i8], align 16
  %a = alloca [2 x ptr], align 8

  ; F1F1F1F1
  ; CHECK: store i32 -235802127, ptr %{{[0-9]+}}
  ; CHECK: call void @__asan_set_shadow_f8(i64 %[[#]], i64 128)
  ; 0xf2f2f2f2f2f2f2f2
  ; CHECK: store i64 -940422246894996750, ptr %[[#]]
  ; 0xf2f2f2f2f2f2f2f2
  ; CHECK: store i64 -940422246894996750, ptr %[[#]]

  call void @llvm.lifetime.start.p0(i64 1024, ptr %x)
  ; CHECK: call void @__asan_set_shadow_00(i64 %{{[0-9]+}}, i64 128)
  ; CHECK-NEXT: call void @llvm.lifetime.start

  %ai = getelementptr inbounds [2 x ptr], ptr %a, i64 0, i64 %i
  store ptr %x, ptr %ai, align 8
  ; CHECK: store ptr

  call void @llvm.lifetime.end.p0(i64 1024, ptr %x)
  ; CHECK: call void @__asan_set_shadow_f8(i64 %{{[0-9]+}}, i64 128)
  ; CHECK-NEXT: call void @llvm.lifetime.end

  ret void
  ; CHECK: call void @__asan_set_shadow_00(i64 %{{[0-9]+}}, i64 148)
  ; CHECK: store i16 0, ptr %[[#]], align 1
  ; CHECK-NEXT: ret void
}

define void @zero_sized(i64 %a) #0 {
; CHECK-LABEL: define void @zero_sized(i64 %a)

entry:
  %a.addr = alloca i64, align 8
  %b = alloca [0 x i8], align 1
  store i64 %a, ptr %a.addr, align 8

  call void @llvm.lifetime.start.p0(i64 0, ptr %b) #2
  ; CHECK: call void @llvm.lifetime.start

  call void @llvm.lifetime.end.p0(i64 0, ptr %b) #2
  ; CHECK: call void @llvm.lifetime.end

  ret void
  ; CHECK-NEXT: ret void
}
