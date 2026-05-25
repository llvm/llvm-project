// Verify that `-d` emits a non-empty depfile that always names the primary
// input file, even when the input has no `include` directives. A depfile
// containing only `<output>:` (no deps) round-trips through CMake's
// `cmake_transform_depfile` step to a 0-byte file, which Ninja treats as a
// missing depfile and unconditionally re-runs the edge.

// RUN: llvm-tblgen -print-records %s -o %t.out -d %t.d
// RUN: FileCheck %s --input-file=%t.d

// CHECK: {{.*}}depfile.td.tmp.out: {{.*}}depfile.td

// When the input is stdin (`-`), there is no input path to emit, so the
// depfile lists only the output target with no dependencies.
// RUN: llvm-tblgen -print-records - -o %t.stdin.out -d %t.stdin.d < %s
// RUN: FileCheck %s --check-prefix=STDIN --input-file=%t.stdin.d

// STDIN: {{.*}}depfile.td.tmp.stdin.out:{{$}}

def Empty;
