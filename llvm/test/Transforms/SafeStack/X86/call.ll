; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; no arrays / no nested arrays
; Requires no protector.

define void @foo(i8* %a) nounwind uwtable safestack {
entry:
  ; CHECK-LABEL: define void @foo(
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8*, i8** %a.addr, align 8
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

declare i32 @printf(i8*, ...)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @call_memset(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @llvm.memset.p0i8.i64(i8* %arraydecay, i8 1, i64 %len, i32 1, i1 false)
  ret void
}

define void @call_constant_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_memset
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 2
  call void @llvm.memset.p0i8.i64(i8* %arraydecay, i8 1, i64 7, i32 1, i1 false)
  ret void
}

define void @call_constant_overflow_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_overflow_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 7
  call void @llvm.memset.p0i8.i64(i8* %arraydecay, i8 1, i64 5, i32 1, i1 false)
  ret void
}

define void @call_constant_underflow_memset() safestack {
entry:
  ; CHECK-LABEL: define void @call_constant_underflow_memset
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr [10 x i8], [10 x i8]* %q, i32 0, i32 -1
  call void @llvm.memset.p0i8.i64(i8* %arraydecay, i8 1, i64 3, i32 1, i1 false)
  ret void
}

; Readnone nocapture -> safe
define void @call_readnone(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @readnone(i8* %arraydecay)
  ret void
}

; Arg0 is readnone, arg1 is not. Pass alloca ptr as arg0 -> safe
define void @call_readnone0_0(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone0_0
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @readnone0(i8* %arraydecay, i8* zeroinitializer)
  ret void
}

; Arg0 is readnone, arg1 is not. Pass alloca ptr as arg1 -> unsafe
define void @call_readnone0_1(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readnone0_1
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @readnone0(i8 *zeroinitializer, i8* %arraydecay)
  ret void
}

; Readonly nocapture -> unsafe
define void @call_readonly(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readonly
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @readonly(i8* %arraydecay)
  ret void
}

; Readonly nocapture -> unsafe
define void @call_arg_readonly(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_arg_readonly
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @arg_readonly(i8* %arraydecay)
  ret void
}

; Readwrite nocapture -> unsafe
define void @call_readwrite(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_readwrite
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @readwrite(i8* %arraydecay)
  ret void
}

; Captures the argument -> unsafe
define void @call_capture(i64 %len) safestack {
entry:
  ; CHECK-LABEL: define void @call_capture
  ; CHECK: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
  %q = alloca [10 x i8], align 1
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %q, i32 0, i32 0
  call void @capture(i8* %arraydecay)
  ret void
}

; Lifetime intrinsics are always safe.
define void @call_lifetime(i32* %p) {
  ; CHECK-LABEL: define void @call_lifetime
  ; CHECK-NOT: @__safestack_unsafe_stack_ptr
  ; CHECK: ret void
entry:
  %q = alloca [100 x i8], align 16
  %0 = bitcast [100 x i8]* %q to i8*
  call void @llvm.lifetime.start.p0i8(i64 100, i8* %0)
  call void @llvm.lifetime.end.p0i8(i64 100, i8* %0)
  ret void
}

declare void @readonly(i8* nocapture) readonly
declare void @arg_readonly(i8* readonly nocapture)
declare void @readwrite(i8* nocapture)
declare void @capture(i8* readnone) readnone

declare void @readnone(i8* nocapture) readnone
declare void @readnone0(i8* nocapture readnone, i8* nocapture)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind argmemonly

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind argmemonly
