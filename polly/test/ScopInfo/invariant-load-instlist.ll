; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s

; The load is a required invariant load and at the same time used in a store.
; Polly used to add two MemoryAccesses for it which caused an assertion to fail.
; llvm.org/PR48059

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.27.29112"

@b = external dso_local global i32, align 4
@c = external dso_local global ptr, align 8
@a = external dso_local local_unnamed_addr global ptr, align 8

define void @func() {
for.cond1.preheader.preheader:
  br label %for.end12

for.end12:
  br i1 undef, label %for.end12.1, label %for.end12

for.end12.1:
  %0 = phi ptr [ %1, %for.end12.1 ], [ undef, %for.end12 ]
  %storemerge26.1 = phi i32 [ %inc14.1, %for.end12.1 ], [ 0, %for.end12 ]
  %1 = load ptr, ptr @c, align 8
  store i32 0, ptr %1, align 4
  %inc14.1 = add nuw nsw i32 %storemerge26.1, 1
  %exitcond.1.not = icmp eq i32 %inc14.1, 35
  br i1 %exitcond.1.not, label %for.inc16.1, label %for.end12.1

for.inc16.1:
  ret void
}
