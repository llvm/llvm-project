// RUN: %clang --target=x86_64-pc-linux -g -gsplit-dwarf -c %s -o %t.o
// RUN: rm %t.dwo
// RUN: %lldb %t.o -o "br set -n main" -o exit 2>&1 | FileCheck %s

// CHECK: warning: {{.*}} unable to locate separate debug file (dwo, dwp). Debugging will be degraded.

int main() { return 47; }
