; REQUIRES: x86-registered-target
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o /dev/null -print-mir2vec-vocab %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_zero_vocab.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-ZERO-DIM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_invalid_vocab.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ENTITIES
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_inconsistent_dims.json %s 2>&1 | FileCheck %s --check-prefix=CHECK-INCONSISTENT-DIMS

define dso_local void @test() {
  entry:
    ret void
}

; CHECK-INVALID: MIR2Vec Vocabulary Printer: Failed to get vocabulary - MIR2Vec vocabulary file path not specified; set it using --mir2vec-vocab-path
; CHECK-ZERO-DIM: MIR2Vec Vocabulary Printer: Failed to get vocabulary - Dimension of 'Opcodes' section of the vocabulary is zero
; CHECK-NO-ENTITIES: MIR2Vec Vocabulary Printer: Failed to get vocabulary - Missing 'Opcodes' section in vocabulary file
; CHECK-INCONSISTENT-DIMS: MIR2Vec Vocabulary Printer: Failed to get vocabulary - All vectors in the 'Opcodes' section of the vocabulary are not of the same dimension
