; RUN: opt -S -aarch64-stack-tagging -stack-tagging-use-stack-safety=0 %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-arm-unknown-eabi"

declare void @use8(i8*)

define  void @f(i1 %cond) local_unnamed_addr sanitize_memtag {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %a)
  call void @use8(i8* %a)
; CHECK: call void @llvm.aarch64.settag(i8* %a.tag, i64 48)
  br i1 %cond, label %next0, label %next1

next0:
; CHECK-LABEL: next0:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit0

exit0:
; CHECK-LABEL: exit0:
; CHECK-NOT: call void @llvm.aarch64.settag
  ret void

next1:
; CHECK-LABEL: next1:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

exit1:
; CHECK-LABEL: exit1:
; CHECK-NOT: call void @llvm.aarch64.settag
  ret void
}

define  void @diamond(i1 %cond) local_unnamed_addr sanitize_memtag {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %a)
  call void @use8(i8* %a)
; CHECK: call void @llvm.aarch64.settag(i8* %a.tag, i64 48)
  br i1 %cond, label %next0, label %next1

next0:
; CHECK-LABEL: next0:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next1:
; CHECK-LABEL: next1:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

exit1:
; CHECK-LABEL: exit1:
; CHECK-NOT: call void @llvm.aarch64.settag
  ret void
}

define  void @diamond_nocover(i1 %cond) local_unnamed_addr sanitize_memtag {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %a)
  call void @use8(i8* %a)
; CHECK: call void @llvm.aarch64.settag(i8* %a.tag, i64 48)
  br i1 %cond, label %next0, label %next1

next0:
; CHECK-LABEL: next0:
; CHECK-NOT: llvm.lifetime.end
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next1:
; CHECK-LABEL: next1:
; CHECK-NOT: llvm.lifetime.end
  br label %exit1

exit1:
; CHECK-LABEL: exit1:
; CHECK: call void @llvm.aarch64.settag
  ret void
}

define  void @diamond3(i1 %cond, i1 %cond1) local_unnamed_addr sanitize_memtag {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %a)
  call void @use8(i8* %a)
; CHECK: call void @llvm.aarch64.settag(i8* %a.tag, i64 48)
  br i1 %cond, label %next0, label %start1

start1:
  br i1 %cond1, label %next1, label %next2

next0:
; CHECK-LABEL: next0:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next1:
; CHECK-LABEL: next1:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next2:
; CHECK-LABEL: next2:
; CHECK: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

exit1:
; CHECK-LABEL: exit1:
; CHECK-NOT: call void @llvm.aarch64.settag
  ret void
}

define  void @diamond3_nocover(i1 %cond, i1 %cond1) local_unnamed_addr sanitize_memtag {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %a)
  call void @use8(i8* %a)
; CHECK: call void @llvm.aarch64.settag(i8* %a.tag, i64 48)
  br i1 %cond, label %next0, label %start1

start1:
  br i1 %cond1, label %next1, label %next2

next0:
; CHECK-LABEL: next0:
; CHECK-NOT: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next1:
; CHECK-LABEL: next1:
; CHECK-NOT: call void @llvm.aarch64.settag
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %a)
  br label %exit1

next2:
; CHECK-LABEL: next2:
; CHECK-NOT: call void @llvm.aarch64.settag
  br label %exit1

exit1:
; CHECK-LABEL: exit1:
; CHECK: call void @llvm.aarch64.settag
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
