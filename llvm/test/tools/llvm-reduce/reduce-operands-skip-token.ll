; Check token values are correctly handled by operands-skip

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-skip --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=RESULT %s < %t

; INTERESTING-LABEL: define void @tokens(
; INTERESTING: call void @llvm.token.consumer

; RESULT-LABEL: define void @tokens(ptr %ptr) {
; RESULT-NEXT: %token = call token @llvm.token.producer()
; RESULT-NEXT:store i32 0, ptr %ptr, align 4
; RESULT-NEXT:%chain = call token @llvm.token.chain(token %token)
; RESULT-NEXT:call void @llvm.token.consumer(token %token)
; RESULT-NEXT:store i32 1, ptr %ptr, align 4
; RESULT-NEXT:ret void
define void @tokens(ptr %ptr) {
  %token = call token @llvm.token.producer()
  store i32 0, ptr %ptr
  %chain = call token @llvm.token.chain(token %token)
  call void @llvm.token.consumer(token %chain) ; -> rewrite to use %token directly
  store i32 1, ptr %ptr
  ret void
}

declare token @llvm.token.producer()
declare token @llvm.token.chain(token)
declare void @llvm.token.consumer(token)
