// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -path-equivalence=/tmp,%S /dev/null | FileCheck %s -allow-empty -check-prefix=FAKE-FILE-STDOUT
// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -path-equivalence=/tmp,%S /dev/null 2>&1 | FileCheck %s -check-prefix=FAKE-FILE-STDERR
// RUN: not llvm-cov report %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -format=html
// RUN: not llvm-cov export %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -format=html

// FAKE-FILE-STDOUT-NOT: warning: The file '{{.*}}' isn't covered.
// FAKE-FILE-STDERR: warning: The file '{{.*}}' isn't covered.

// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -path-equivalence=/tmp,%S -name ".*" /dev/null | FileCheck %s -allow-empty -check-prefix=FAKE-FUNC-STDOUT
// RUN: llvm-cov show %S/Inputs/prevent_false_instantiations.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata -path-equivalence=/tmp,%S -name-regex ".*" /dev/null 2>&1 | FileCheck %s -check-prefix=FAKE-FUNC-STDERR

// FAKE-FUNC-STDOUT-NOT: warning: Could not read coverage for '{{.*}}'.
// FAKE-FUNC-STDERR: Could not read coverage for '{{.*}}'.

// RUN: not llvm-cov report %S/Inputs/malformedRegions.covmapping -instr-profile %S/Inputs/elf_binary_comdat.profdata 2>&1 | FileCheck %s -check-prefix=MALFORMED-REGION
// MALFORMED-REGION: malformedRegions.covmapping: Failed to load coverage: Malformed coverage data
