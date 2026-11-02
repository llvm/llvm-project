// Test that lldb reports an error when debug map object files are missing,
// instead of silently dropping their debug info, and reports it once per
// module.
// REQUIRES: system-darwin
// RUN: %clang_host %s -g -c -o %t.o
// RUN: %clang_host %s -g -c -DFOO -o %t.foo.o
// RUN: %clang_host %t.o -g -o %t
// RUN: %clang_host %t.o %t.foo.o -g -o %t.two
// RUN: rm %t.o %t.foo.o
// RUN: %lldb %t -o "breakpoint set -f %s -l 28" -o exit 2>&1 | FileCheck %s
// RUN: %lldb %t.two -o "breakpoint set -f %s -l 28" \
// RUN:   -o "image dump separate-debug-info --errors-only" -o exit 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TWO

// CHECK: error: {{.*}}.o" containing debug info does not exist, debug info will not be loaded

// TWO: error: {{.*}} 2 of 2 debug map object files do not exist: debug info cannot be loaded for them
// TWO-NOT: error:
// TWO: (lldb) image dump separate-debug-info --errors-only
// TWO: Type: "oso"
// TWO: E debug map object file "{{.*}}.tmp.o" containing debug info does not exist
// TWO: E debug map object file "{{.*}}.tmp.foo.o" containing debug info does not exist
// TWO-NOT: error:

#ifdef FOO
int foo() { return 1; }
#else
int main() { return 0; }
#endif
