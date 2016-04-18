// Checks that function instantiations don't go to a wrong file.

// CHECK-NOT: {{_Z5func[1,2]v}}

// RUN: llvm-profdata merge %S/Inputs/prevent_false_instantiations.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck %s

#define DO_SOMETHING() \
  do {                 \
  } while (0)
