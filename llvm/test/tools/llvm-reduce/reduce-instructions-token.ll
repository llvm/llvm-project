; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; CHECK-LABEL: define void @tokens(
; INTERESTING: store i32 0
; INTERESTING: call void @llvm.token.consumer

; RESULT: %token = call token @llvm.token.producer()
; RESULT-NEXT: store i32 0, ptr %ptr, align 4
; RESULT-NEXT: call void @llvm.token.consumer(token %token)
; RESULT-NEXT: ret void
define void @tokens(ptr %ptr) {
  %token = call token @llvm.token.producer()
  store i32 0, ptr %ptr
  call void @llvm.token.consumer(token %token)
  store i32 1, ptr %ptr
  ret void
}

declare token @llvm.token.producer()
declare void @llvm.token.consumer(token)
