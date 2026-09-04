; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t
; RUN: llc -mtriple=nvptx64-nvidia-cuda -opt-bisect-limit=%(line-1) -O3 %s -o %t

;; This test is intended to verify that we don't crash when -opt-bisect-limit
;; is used in conjunction with lifetime markers. Previously, later passes
;; would not handle these intructions correctly and relied on earlier passes
;; to remove them.

declare void @bar(ptr)

define void @foo() {
  %p = alloca i32
  call void @llvm.lifetime.start(ptr %p)
  call void @bar(ptr %p)
  call void @llvm.lifetime.end(ptr %p)
  ret void
}
