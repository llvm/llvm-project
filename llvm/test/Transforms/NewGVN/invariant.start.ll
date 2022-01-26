; Test to make sure llvm.invariant.start calls are not treated as clobbers.
; RUN: opt < %s -passes=newgvn -S | FileCheck %s


declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind

; We forward store to the load across the invariant.start intrinsic
define i8 @forward_store() {
; CHECK-LABEL: @forward_store
; CHECK: call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
; CHECK-NOT: load
; CHECK: ret i8 0
  %a = alloca i8
  store i8 0, i8* %a
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
  %r = load i8, i8* %a
  ret i8 %r
}

declare i8 @dummy(i8* nocapture) nounwind readonly

; We forward store to the load in the non-local analysis case,
; i.e. invariant.start is in another basic block.
define i8 @forward_store_nonlocal(i1 %cond) {
; CHECK-LABEL: forward_store_nonlocal
; CHECK: call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
; CHECK: ret i8 0
; CHECK: ret i8 %val
  %a = alloca i8
  store i8 0, i8* %a
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
  br i1 %cond, label %loadblock, label %exit

loadblock:
  %r = load i8, i8* %a
  ret i8 %r

exit:
  %val = call i8 @dummy(i8* %a)
  ret i8 %val
}

; We should not value forward %foo to the invariant.end corresponding to %bar.
define i8 @forward_store1() {
; CHECK-LABEL: forward_store1
; CHECK: %foo = call {}* @llvm.invariant.start.p0i8
; CHECK-NOT: load
; CHECK: %bar = call {}* @llvm.invariant.start.p0i8
; CHECK: call void @llvm.invariant.end.p0i8({}* %bar, i64 1, i8* %a)
; CHECK: ret i8 0
  %a = alloca i8
  store i8 0, i8* %a
  %foo = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
  %r = load i8, i8* %a
  %bar = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %a)
  call void @llvm.invariant.end.p0i8({}* %bar, i64 1, i8* %a)
  ret i8 %r
}
