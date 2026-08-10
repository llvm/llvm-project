// To create the covmapping for this file, copy this file to /tmp/dots/,
// cd into /tmp/dots, and pass "../dots/double_dots.c" to the compiler. Use
// llvm-cov convert-for-testing to extract the covmapping.

// RUN: llvm-profdata merge %S/Inputs/double_dots.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/double_dots.covmapping -instr-profile=%t.profdata -o %t.dir
// RUN: FileCheck -input-file=%t.dir/index.txt %s
// RUN: llvm-cov show -format=html %S/Inputs/double_dots.covmapping -instr-profile=%t.profdata -o %t.dir
// RUN: FileCheck -input-file=%t.dir/index.html %s

// CHECK-NOT: coverage{{.*}}dots{{.*}}..{{.*}}dots

int main() {}

// Re-purpose this file to test that we use relative paths when creating
// report indices:

// RUN: FileCheck -check-prefix=REL-INDEX -input-file %t.dir/index.txt %s
// REL-INDEX-NOT: %t.dir

// Check that we get the right error when writing to an invalid path:

// RUN: not llvm-cov show %S/Inputs/double_dots.covmapping -instr-profile=%t.profdata -o /dev/null 2>&1 | FileCheck %s -check-prefix=ERROR-MESSAGE
// ERROR-MESSAGE: error: {{.*}}: could not create index file!
