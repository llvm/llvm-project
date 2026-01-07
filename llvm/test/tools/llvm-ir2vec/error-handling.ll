; Test error handling and input validation for llvm-ir2vec tool

; RUN: not llvm-ir2vec embeddings %s 2>&1 | FileCheck %s -check-prefix=CHECK-NO-VOCAB
; RUN: not llvm-ir2vec embeddings --function=nonexistent --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s -check-prefix=CHECK-FUNC-NOT-FOUND

; Simple test function for valid IR
define i32 @test_func(i32 %a) {
entry:
  ret i32 %a
}

; CHECK-NO-VOCAB: error: IR2Vec vocabulary file path not specified; You may need to set it using --ir2vec-vocab-path
; CHECK-FUNC-NOT-FOUND: error: Function 'nonexistent' not found
