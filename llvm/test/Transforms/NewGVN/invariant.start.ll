; Test to make sure llvm.invariant.start calls are not treated as clobbers.
; RUN: opt < %s -passes=newgvn -S | FileCheck %s


declare ptr @llvm.invariant.start.p0(i64, ptr nocapture) nounwind readonly
declare void @llvm.invariant.end.p0(ptr, i64, ptr nocapture) nounwind

; We forward store to the load across the invariant.start intrinsic
define i8 @forward_store() {
; CHECK-LABEL: @forward_store
; CHECK: call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
; CHECK-NOT: load
; CHECK: ret i8 0
  %a = alloca i8
  store i8 0, ptr %a
  %i = call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
  %r = load i8, ptr %a
  ret i8 %r
}

declare i8 @dummy(ptr nocapture) nounwind readonly

; We forward store to the load in the non-local analysis case,
; i.e. invariant.start is in another basic block.
define i8 @forward_store_nonlocal(i1 %cond) {
; CHECK-LABEL: forward_store_nonlocal
; CHECK: call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
; CHECK: ret i8 0
; CHECK: ret i8 %val
  %a = alloca i8
  store i8 0, ptr %a
  %i = call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
  br i1 %cond, label %loadblock, label %exit

loadblock:
  %r = load i8, ptr %a
  ret i8 %r

exit:
  %val = call i8 @dummy(ptr %a)
  ret i8 %val
}

; We should not value forward %foo to the invariant.end corresponding to %bar.
define i8 @forward_store1() {
; CHECK-LABEL: forward_store1
; CHECK: %foo = call ptr @llvm.invariant.start.p0
; CHECK-NOT: load
; CHECK: %bar = call ptr @llvm.invariant.start.p0
; CHECK: call void @llvm.invariant.end.p0(ptr %bar, i64 1, ptr %a)
; CHECK: ret i8 0
  %a = alloca i8
  store i8 0, ptr %a
  %foo = call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
  %r = load i8, ptr %a
  %bar = call ptr @llvm.invariant.start.p0(i64 1, ptr %a)
  call void @llvm.invariant.end.p0(ptr %bar, i64 1, ptr %a)
  ret i8 %r
}
