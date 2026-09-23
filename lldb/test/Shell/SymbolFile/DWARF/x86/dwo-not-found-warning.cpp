// RUN: %clangxx --target=x86_64-pc-linux -g -gsplit-dwarf -c %s -o %t.o
// RUN: rm %t.dwo
// RUN: lldb-test dwo-diagnostic-suffix > %t.expected
// RUN: %lldb %t.o -o "br set -n main" -o exit > %t.output 2>&1
// RUN: sed -n 's/^warning: .* unable to locate separate debug file (dwo, dwp)\. //p' %t.output > %t.actual
// RUN: diff -u --strip-trailing-cr %t.expected %t.actual

int main() { return 47; }
