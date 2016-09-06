// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -filename-equivalence /dev/null | FileCheck %s -allow-empty -check-prefix=FAKE-FILE-STDOUT
// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -filename-equivalence /dev/null 2>&1 | FileCheck %s -check-prefix=FAKE-FILE-STDERR

// FAKE-FILE-STDOUT-NOT: warning: The file '{{.*}}' isn't covered.
// FAKE-FILE-STDERR: warning: The file '{{.*}}' isn't covered.

// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -filename-equivalence -name ".*" /dev/null | FileCheck %s -allow-empty -check-prefix=FAKE-FUNC-STDOUT
// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -filename-equivalence -name-regex ".*" /dev/null 2>&1 | FileCheck %s -check-prefix=FAKE-FUNC-STDERR

// FAKE-FUNC-STDOUT-NOT: warning: Could not read coverage for '{{.*}}'.
// FAKE-FUNC-STDERR: Could not read coverage for '{{.*}}'.
