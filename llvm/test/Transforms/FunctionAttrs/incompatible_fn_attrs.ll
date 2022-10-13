; RUN: opt -S -o - -function-attrs %s | FileCheck %s
; RUN: opt -S -o - -passes=function-attrs %s | FileCheck %s

; Verify we remove argmemonly/inaccessiblememonly/inaccessiblemem_or_argmemonly
; function attributes when we derive readnone.

; Function Attrs: argmemonly
define ptr @given_argmem_infer_readnone(ptr %p) #0 {
; CHECK: define ptr @given_argmem_infer_readnone(ptr readnone returned %p) #0 {
entry:
  ret ptr %p
}

; Function Attrs: inaccessiblememonly
define ptr @given_inaccessible_infer_readnone(ptr %p) #1 {
; CHECK: define ptr @given_inaccessible_infer_readnone(ptr readnone returned %p) #0 {
entry:
  ret ptr %p
}

; Function Attrs: inaccessiblemem_or_argmemonly
define ptr @given_inaccessible_or_argmem_infer_readnone(ptr %p) #2 {
; CHECK: define ptr @given_inaccessible_or_argmem_infer_readnone(ptr readnone returned %p) #0 {
entry:
  ret ptr %p
}

attributes #0 = { argmemonly }
attributes #1 = { inaccessiblememonly }
attributes #2 = { inaccessiblemem_or_argmemonly }
; CHECK: attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn }
; CHECK-NOT: attributes
