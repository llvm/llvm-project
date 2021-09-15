// RUN: llvm-profdata merge %S/Inputs/path_equivalence.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/path_equivalence.covmapping -instr-profile=%t.profdata -path-equivalence=/tmp,%S | FileCheck %s
// RUN: llvm-cov show %S/Inputs/path_equivalence.covmapping -instr-profile=%t.profdata -path-equivalence=/tmp/,%S/ | FileCheck %s
int main() {} // CHECK: [[@LINE]]|      1|int main() {}

// RUN: not llvm-cov show --instr-profile=/dev/null -path-equivalence=foo /dev/null 2>&1 | FileCheck --check-prefix=INVALID %s
// INVALID: error: -path-equivalence: invalid argument 'foo', must be in format 'from,to'
