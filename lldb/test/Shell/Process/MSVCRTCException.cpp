// clang-format off

// Test that lldb prints MSVC's runtime checks exceptions as stop reasons.

// REQUIRES: msvc

// RUN: %msvc_cl /nologo /Od /Zi /MDd /RTC1 -o %t.exe %s
// RUN: %lldb -f %t.exe -b -o 'r' 2>&1 | FileCheck %s
// CHECK: thread #1, stop reason = Run-time check failure: The variable 'x' is being used without being initialized.

#include <iostream>

int main() {
    int x;
    printf("%d\n", x);
}