; RUN: opt -inline -S < %s | FileCheck %s

; The callee guarantees that the pointer argument is nonnull and dereferenceable.
; That information should transfer to the caller.

define i32 @callee(i32* dereferenceable(32) %t1) {
; CHECK-LABEL: @callee(i32* dereferenceable(32) %t1)
; CHECK-NEXT:    [[T2:%.*]] = load i32, i32* %t1
; CHECK-NEXT:    ret i32 [[T2]]
;
  %t2 = load i32, i32* %t1
  ret i32 %t2
}

; Add a nonnull assumption. 
; The caller argument could be known nonnull and dereferenceable(32).

define i32 @caller1(i32* %t1) {
; CHECK-LABEL: @caller1(i32* %t1)
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne i32* %t1, null
; CHECK-NEXT:    call void @llvm.assume(i1 [[TMP1]])
; CHECK-NEXT:    [[T2_I:%.*]] = load i32, i32* %t1
; CHECK-NEXT:    ret i32 [[T2_I]]
;
  %t2 = tail call i32 @callee(i32* dereferenceable(32) %t1)
  ret i32 %t2
}

; Don't add a nonnull assumption if it's redundant.
; The caller argument is nonnull, but that can be explicit.
; The dereferenceable amount could be increased.

define i32 @caller2(i32* dereferenceable(31) %t1) {
; CHECK-LABEL: @caller2(i32* dereferenceable(31) %t1)
; CHECK-NEXT:    [[T2_I:%.*]] = load i32, i32* %t1
; CHECK-NEXT:    ret i32 [[T2_I]]
;
  %t2 = tail call i32 @callee(i32* dereferenceable(32) %t1)
  ret i32 %t2
}

; Don't add a nonnull assumption if it's redundant.
; The caller argument is nonnull, but that can be explicit.
; Make sure that we don't propagate a smaller dereferenceable amount.

define i32 @caller3(i32* dereferenceable(33) %t1) {
; CHECK-LABEL: @caller3(i32* dereferenceable(33) %t1)
; CHECK-NEXT:    [[T2_I:%.*]] = load i32, i32* %t1
; CHECK-NEXT:    ret i32 [[T2_I]]
;
  %t2 = tail call i32 @callee(i32* dereferenceable(32) %t1)
  ret i32 %t2
}

