// Checks that function instantiations don't go to a wrong file.

// INSTANTIATION-NOT: {{_Z5func[1,2]v}}
// NAN-NOT: {{[ \t]+}}nan%

// RUN: llvm-profdata merge %S/Inputs/prevent_false_instantiations.proftext -o %t.profdata
// RUN: llvm-cov show -format text %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %t.profdata -path-equivalence=/tmp/false_instantiations/./,%S %s | FileCheck %s -check-prefix=INSTANTIATION
// RUN: llvm-cov report %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %t.profdata | FileCheck %s -check-prefix=NAN

#define DO_SOMETHING() \
  do {                 \
  } while (0)
