// Tests that we can evaluate functions that Clang
// classifies as having clang::Linkage::UniqueExternal
// linkage. In this case, a function whose argument
// is not legally usable outside this TU.

// XFAIL: target-windows

// RUN: %build %s -o %t
// RUN: %lldb %t -o run -o "expression func(a)" -o exit | FileCheck %s

// CHECK: expression func(a)
// CHECK: (int) $0 = 15

namespace {
struct InAnon {};
} // namespace

int func(InAnon a) { return 15; }

int main() {
  InAnon a;
  __builtin_debugtrap();
  return func(a);
}
