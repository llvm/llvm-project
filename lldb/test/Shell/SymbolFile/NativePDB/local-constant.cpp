// clang-format off
// REQUIRES: msvc

// Test that we can display local S_CONSTANT records.
// MSVC emits S_CONSTANT for static const locals; clang-cl does not.

// RUN: %build --compiler=msvc --nodefaultlib -o %t.exe -- %s
// RUN: %lldb  -o "settings set stop-line-count-after 0" \
// RUN:   -f %t.exe -s %p/Inputs/local-constant.lldbinit 2>&1 | FileCheck %s

int main() {
  static const int kConstant = 42;
  return kConstant;
}

// CHECK: (const int) {{.*}}kConstant = 42
