; Check no invalid reduction caused by introducing a token typed
; function argument

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=RESULT %s < %t

; INTERESTING-LABEL: define void @tokens(
; INTERESTING: call void @llvm.token.consumer

; RESULT-LABEL: define void @tokens(ptr %local.ptr) {
; RESULT-NEXT: %local.ptr1 = alloca i32, align 4
; RESULT-NEXT: %token = call token @llvm.token.producer()
; RESULT-NEXT: store i32 0, ptr %local.ptr, align 4
; RESULT-NEXT: call void @llvm.token.consumer(token %token)
; RESULT-NEXT: store i32 1, ptr %local.ptr, align 4
; RESULT-NEXT: ret void
define void @tokens() {
  %local.ptr = alloca i32
  %token = call token @llvm.token.producer()
  store i32 0, ptr %local.ptr
  call void @llvm.token.consumer(token %token)
  store i32 1, ptr %local.ptr
  ret void
}

declare token @llvm.token.producer()
declare void @llvm.token.consumer(token)
