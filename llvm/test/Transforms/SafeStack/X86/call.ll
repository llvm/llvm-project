; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; no arrays / no nested arrays
; Requires no protector.

define void @foo(ptr %a) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define void @foo(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

declare i32 @printf(ptr, ...)

define void @call_memset(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @llvm.memset.p0.i64(ptr %q, i8 1, i64 %len, i1 false)
  ret void
}

define void @call_constant_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_memset
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], ptr %q, i32 0, i32 2
  call void @llvm.memset.p0.i64(ptr %arraydecay, i8 1, i64 7, i1 false)
  ret void
}

define void @call_constant_overflow_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_overflow_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], ptr %q, i32 0, i32 7
  call void @llvm.memset.p0.i64(ptr %arraydecay, i8 1, i64 5, i1 false)
  ret void
}

define void @call_constant_underflow_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_underflow_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr [10 x i8], ptr %q, i32 0, i32 -1
  call void @llvm.memset.p0.i64(ptr %arraydecay, i8 1, i64 3, i1 false)
  ret void
}

; Readnone nocapture -> safe
define void @call_readnone(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @readnone(ptr %q)
  ret void
}

; Arg0 is readnone, arg1 is not. Pass alloca ptr as arg0 -> safe
define void @call_readnone0_0(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone0_0
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @readnone0(ptr %q, ptr zeroinitializer)
  ret void
}

; Arg0 is readnone, arg1 is not. Pass alloca ptr as arg1 -> unsafe
define void @call_readnone0_1(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone0_1
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @readnone0(ptr zeroinitializer, ptr %q)
  ret void
}

; Readonly nocapture -> unsafe
define void @call_readonly(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readonly
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @readonly(ptr %q)
  ret void
}

; Readonly nocapture -> unsafe
define void @call_arg_readonly(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_arg_readonly
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @arg_readonly(ptr %q)
  ret void
}

; Readwrite nocapture -> unsafe
define void @call_readwrite(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readwrite
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @readwrite(ptr %q)
  ret void
}

; Captures the argument -> unsafe
define void @call_capture(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_capture
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  call void @capture(ptr %q)
  ret void
}

; Lifetime intrinsics are always safe.
define void @call_lifetime(ptr %p) {
  ; CHECK-LABEL: define void @call_lifetime
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
entry:
  %q = alloca [100 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 100, ptr %q)
  call void @llvm.lifetime.end.p0(i64 100, ptr %q)
  ret void
}

declare void @readonly(ptr nocapture) readonly
declare void @arg_readonly(ptr readonly nocapture)
declare void @readwrite(ptr nocapture)
declare void @capture(ptr readnone) readnone

declare void @readnone(ptr nocapture) readnone
declare void @readnone0(ptr nocapture readnone, ptr nocapture)

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind argmemonly

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind argmemonly
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind argmemonly
