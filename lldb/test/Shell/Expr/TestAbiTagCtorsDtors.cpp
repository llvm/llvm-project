// Tests that we can call abi-tagged constructors/destructors.

// RUN: %build %s -o %t
// RUN: %lldb %t -o run \
// RUN:          -o "expression sinkTagged(getTagged())" \
// RUN:          -o "expression Tagged()" \
// RUN:          -o exit | FileCheck %s

// CHECK: expression sinkTagged(getTagged())
// CHECK: expression Tagged()

struct Tagged {
  [[gnu::abi_tag("Test", "CtorTag")]] [[gnu::abi_tag("v1")]] Tagged() = default;
  [[gnu::abi_tag("Test", "DtorTag")]] [[gnu::abi_tag("v1")]] ~Tagged() {}

  int mem = 15;
};

Tagged getTagged() { return Tagged(); }
void sinkTagged(Tagged t) {}

int main() {
  Tagged t;

  // TODO: is there a more reliable way of triggering destructor call?
  sinkTagged(getTagged());
  __builtin_debugtrap();
}
