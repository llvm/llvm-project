; RUN: opt < %s -passes=sroa -S | FileCheck %s
;
; Make sure that SROA doesn't lose nonnull metadata
; on loads from allocas that get optimized out.

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

; Check that we do basic propagation of nonnull when rewriting.
define ptr @propagate_nonnull(ptr %v) {
; CHECK-LABEL: define ptr @propagate_nonnull(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[A:.*]] = alloca ptr
; CHECK-NEXT:    store ptr %v, ptr %[[A]]
; CHECK-NEXT:    %[[LOAD:.*]] = load volatile ptr, ptr %[[A]], align 8, !nonnull !0
; CHECK-NEXT:    ret ptr %[[LOAD]]
entry:
  %a = alloca [2 x ptr]
  %a.gep1 = getelementptr [2 x ptr], ptr %a, i32 0, i32 1
  store ptr %v, ptr %a.gep1
  store ptr null, ptr %a
  %load = load volatile ptr, ptr %a.gep1, !nonnull !0
  ret ptr %load
}

define ptr @turn_nonnull_into_assume(ptr %arg) {
; CHECK-LABEL: define ptr @turn_nonnull_into_assume(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[RETURN:.*]] = load ptr, ptr %arg, align 8
; CHECK-NEXT:    %[[ASSUME:.*]] = icmp ne ptr %[[RETURN]], null
; CHECK-NEXT:    call void @llvm.assume(i1 %[[ASSUME]])
; CHECK-NEXT:    ret ptr %[[RETURN]]
entry:
  %buf = alloca ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %buf, ptr align 8 %arg, i64 8, i1 false)
  %ret = load ptr, ptr %buf, align 8, !nonnull !0
  ret ptr %ret
}

; Make sure we properly handle the !nonnull attribute when we convert
; a pointer load to an integer load.
; FIXME: While this doesn't do anythnig actively harmful today, it really
; should propagate the !nonnull metadata to range metadata. The irony is, it
; *does* initially, but then we lose that !range metadata before we finish
; SROA.
define ptr @propagate_nonnull_to_int() {
; CHECK-LABEL: define ptr @propagate_nonnull_to_int(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[A:.*]] = alloca ptr
; CHECK-NEXT:    store ptr inttoptr (i64 42 to ptr), ptr %[[A]]
; CHECK-NEXT:    %[[LOAD:.*]] = load volatile ptr, ptr %[[A]]
; CHECK-NEXT:    ret ptr %[[LOAD]]
entry:
  %a = alloca [2 x ptr]
  %a.gep1 = getelementptr [2 x ptr], ptr %a, i32 0, i32 1
  store i64 42, ptr %a.gep1
  store i64 0, ptr %a
  %load = load volatile ptr, ptr %a.gep1, !nonnull !0
  ret ptr %load
}

; Make sure we properly handle the !nonnull attribute when we convert
; a pointer load to an integer load and immediately promote it to an SSA
; register. This can fail in interesting ways due to the rewrite iteration of
; SROA, resulting in PR32902.
define ptr @propagate_nonnull_to_int_and_promote() {
; CHECK-LABEL: define ptr @propagate_nonnull_to_int_and_promote(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr inttoptr (i64 42 to ptr)
entry:
  %a = alloca [2 x ptr], align 8
  %a.gep1 = getelementptr [2 x ptr], ptr %a, i32 0, i32 1
  store i64 42, ptr %a.gep1
  store i64 0, ptr %a
  %load = load ptr, ptr %a.gep1, align 8, !nonnull !0
  ret ptr %load
}

!0 = !{}
