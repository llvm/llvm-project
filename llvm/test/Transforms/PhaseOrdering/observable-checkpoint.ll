; RUN: opt -passes='default<O2>' -S %s | FileCheck %s
; RUN: opt -passes='default<O3>' -S %s | FileCheck %s
; RUN: opt -passes='lto<O2>' -S %s | FileCheck %s
; RUN: opt -passes='thinlto-pre-link<O2>' -S %s | FileCheck %s
; RUN: opt -passes='thinlto<O2>' -S %s | FileCheck %s
; RUN: opt -passes=attributor -S %s | opt -passes='default<O2>' -S | FileCheck %s

@x = external global i32

declare void @llvm.observable.checkpoint()

; The last store in the defined prefix must survive, even when the rest of
; the function is unconditionally undefined. The intrinsic must not acquire
; willreturn through attribute inference on a mustprogress function.
;
; CHECK-LABEL: define {{.*}}void @defined_prefix(
; CHECK: store i32 42, ptr @x
; CHECK-NEXT: {{(tail )?}}call void @llvm.observable.checkpoint()
; CHECK-NEXT: unreachable
define void @defined_prefix() mustprogress {
  store i32 42, ptr @x
  call void @llvm.observable.checkpoint()
  store i32 43, ptr @x
  unreachable
}

; The same store may be removed when there is no checkpoint.
; CHECK-LABEL: define {{.*}}void @no_checkpoint(
; CHECK-NOT: store
; CHECK: unreachable
define void @no_checkpoint() mustprogress {
  store i32 42, ptr @x
  unreachable
}

; A dereference after a checkpoint must not imply nonnull before it. The
; checkpoint itself has no memory effects on %p, so this also tests that its
; possible termination is respected independently of alias analysis.
;
; CHECK-LABEL: define {{.*}}i32 @null_check(
; CHECK: icmp eq ptr %p, null
; CHECK: br i1
; CHECK: store i32 42, ptr @x
; CHECK: call void @llvm.observable.checkpoint()
; CHECK: load i32, ptr %p
; CHECK: ret i32
define i32 @null_check(ptr %p) mustprogress {
  %null = icmp eq ptr %p, null
  br i1 %null, label %bad, label %cont

bad:
  store i32 42, ptr @x
  br label %cont

cont:
  call void @llvm.observable.checkpoint()
  %v = load i32, ptr %p
  ret i32 %v
}

; A checkpoint only protects the prefix. UB after it still permits removing
; code in the suffix.
;
; CHECK-LABEL: define {{.*}}void @undefined_suffix(
; CHECK: call void @llvm.observable.checkpoint()
; CHECK-NOT: store i32 42
; CHECK: unreachable
define void @undefined_suffix() mustprogress {
  call void @llvm.observable.checkpoint()
  store i32 42, ptr @x
  unreachable
}

; A checkpoint is retained even when no UB is visible in this module. Later
; inlining or LTO may expose UB in the caller.
;
; CHECK-LABEL: define {{.*}}void @checkpoint_wrapper(
; CHECK: call void @llvm.observable.checkpoint()
; CHECK-NEXT: ret void
define void @checkpoint_wrapper() alwaysinline mustprogress {
  call void @llvm.observable.checkpoint()
  ret void
}

; In particular, inlining the wrapper cannot expose the prefix to later UB.
; CHECK-LABEL: define {{.*}}void @inline_checkpoint(
; CHECK: store i32 42, ptr @x
; CHECK-NEXT: {{(tail )?}}call void @llvm.observable.checkpoint()
; CHECK-NEXT: unreachable
define void @inline_checkpoint() mustprogress {
  store i32 42, ptr @x
  call void @checkpoint_wrapper()
  unreachable
}

; Checkpoints do not clobber program memory. Forwarding a stored value is
; still valid when execution continues past the checkpoint.
; CHECK-LABEL: define {{.*}}i32 @no_memory_clobber(
; CHECK: store i32 42, ptr %p
; CHECK: call void @llvm.observable.checkpoint()
; CHECK-NEXT: ret i32 42
define i32 @no_memory_clobber(ptr %p) mustprogress {
  store i32 42, ptr %p
  call void @llvm.observable.checkpoint()
  %v = load i32, ptr %p
  ret i32 %v
}
