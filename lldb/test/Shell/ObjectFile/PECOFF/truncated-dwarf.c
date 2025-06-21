// REQUIRES: target-windows
// RUN: %build --compiler=clang-cl --force-dwarf-symbols --force-ms-link -o %t.exe -- %s
// RUN: %lldb -f %t.exe 2>&1 | FileCheck %s

int main(void) {}

// CHECK: warning: {{.*}} contains 4 DWARF sections with truncated names (.debug_{{[a-z]}}, .debug_{{[a-z]}}, .debug_{{[a-z]}}, .debug_{{[a-z]}})
