// Test that lldb does not report missing debug map object files for shared
// cache images, but still records the error.
// REQUIRES: system-darwin
// RUN: %clang_host %s -g -c -o %t.o
// RUN: %clang_host %t.o -g -o %t
// RUN: rm %t.o
// Set MH_DYLIB_IN_CACHE in the Mach-O header flags.
// RUN: %python -c "import sys; f = open(sys.argv[1], 'r+b'); f.seek(24); flags = int.from_bytes(f.read(4), 'little'); f.seek(24); f.write((flags | 0x80000000).to_bytes(4, 'little'))" %t
// RUN: %lldb %t -o "breakpoint set -f %s -l 16" \
// RUN:   -o "image dump separate-debug-info --errors-only" -o exit 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=error:

// CHECK: (lldb) image dump separate-debug-info --errors-only
// CHECK: E debug map object file "{{.*}}.o" containing debug info does not exist

int main() { return 0; }
