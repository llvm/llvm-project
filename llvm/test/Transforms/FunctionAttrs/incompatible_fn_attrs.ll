; RUN: opt -S -o - -function-attrs %s | FileCheck %s
; RUN: opt -S -o - -passes=function-attrs %s | FileCheck %s

; Verify we remove argmemonly/inaccessiblememonly/inaccessiblemem_or_argmemonly
; function attributes when we derive readnone.

; Function Attrs: argmemonly
define i32* @given_argmem_infer_readnone(i32* %p) #0 {
; CHECK: define i32* @given_argmem_infer_readnone(i32* readnone returned %p) #0 {
entry:
  ret i32* %p
}

; Function Attrs: inaccessiblememonly
define i32* @given_inaccessible_infer_readnone(i32* %p) #1 {
; CHECK: define i32* @given_inaccessible_infer_readnone(i32* readnone returned %p) #0 {
entry:
  ret i32* %p
}

; Function Attrs: inaccessiblemem_or_argmemonly
define i32* @given_inaccessible_or_argmem_infer_readnone(i32* %p) #2 {
; CHECK: define i32* @given_inaccessible_or_argmem_infer_readnone(i32* readnone returned %p) #0 {
entry:
  ret i32* %p
}

attributes #0 = { argmemonly }
attributes #1 = { inaccessiblememonly }
attributes #2 = { inaccessiblemem_or_argmemonly }
; CHECK: attributes #0 = { norecurse nounwind readnone }
; CHECK-NOT: attributes
