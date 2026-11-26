; RUN: opt -passes='print<ir2vec>' -disable-output -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_arg_vocab.json %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<ir2vec>' -ir2vec-kind=flow-aware -disable-output -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_arg_vocab.json %s 2>&1 | FileCheck %s

; Metadata operands are skipped; only the two Variable operands and the callee
; (Function) contribute to the embedding.

define double @f(double %a, double %b) strictfp {
  %r = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.strict") strictfp
  ret double %r
}

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

; CHECK: Instruction vectors:
; CHECK-NEXT: Instruction:   %r = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #{{[0-9]+}} [ 4.20  4.80  5.40 ]
