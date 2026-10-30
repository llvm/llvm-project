// Test that lldb reports an error when a debug map object file is missing,
// instead of silently dropping the debug info for that object file.
// REQUIRES: system-darwin
// RUN: %clang_host %s -g -c -o %t.o
// RUN: %clang_host %t.o -g -o %t
// RUN: rm %t.o
// RUN: %lldb %t -o "breakpoint set -f %s -l 10" -o exit 2>&1 | FileCheck %s

// CHECK: error: {{.*}}.o" containing debug info does not exist, debug info will not be loaded

int main() { return 0; }
