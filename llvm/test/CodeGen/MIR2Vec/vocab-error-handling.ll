; REQUIRES: x86_64-linux
; RUN: not llc -o /dev/null -print-mir2vec-vocab %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
; RUN: not llc -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_zero_vocab.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-ZERO-DIM
; RUN: not llc -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_invalid_vocab.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ENTITIES
; RUN: not llc -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_inconsistent_dims.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-INCONSISTENT-DIMS

define dso_local void @test() {
  entry:
    ret void
}

; CHECK-INVALID: error: MIR2Vec vocabulary file path not specified; set it using --mir2vec-vocab-path
; CHECK-ZERO-DIM: error: Dimension of 'entities' section of the vocabulary is zero
; CHECK-NO-ENTITIES: error: Missing 'entities' section in vocabulary file
; CHECK-INCONSISTENT-DIMS: error: All vectors in the 'entities' section of the vocabulary are not of the same dimension
