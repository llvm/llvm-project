// REQUIRES: system-darwin
//
// Tests the language fall back diagnostic for when we fall back to
// Objective-C++ when stopped in frames with no debug-info.
//
// RUN: %clangxx_host %s -o %t.out
//
// RUN: %lldb %t.out \
// RUN:    -o "b main" \
// RUN:    -o run \
// RUN:    -o "expr --language c++ -- blah" -o quit 2>&1 | FileCheck %s

// CHECK: (lldb) expr
// CHECK: note: Possibly stopped inside system library, so speculatively enabled Objective-C. Ran expression as 'Objective C++'.

int main() {
  int x = 10;
  return x;
}
