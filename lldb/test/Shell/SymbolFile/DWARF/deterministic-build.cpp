// Test that binaries linked deterministically (N_OSO has timestamp 0) can still
// have their object files loaded by lldb. Note that the env var ZERO_AR_DATE
// requires the ld64 linker, which clang invokes by default.
// REQUIRES: system-darwin
// RUN: %clang_host %s -g -c -o %t.o
// RUN: ZERO_AR_DATE=1 %clang_host %t.o -g -o %t
// RUN: %lldb %t -o "breakpoint set -f %s -l 11" -o run -o exit | FileCheck %s
// CHECK: stop reason = breakpoint


int main() { return 0; }
