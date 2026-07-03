// RUN: mlir-translate --export-smtlib %s --split-input-file --verify-diagnostics

smt.solver () : () -> () {
  %0 = smt.bv.constant #smt.bv<5> : !smt.bv<16>
  // expected-error @below {{operation not supported for SMTLIB emission}}
  %1 = smt.bv2int %0 signed : !smt.bv<16>
}
