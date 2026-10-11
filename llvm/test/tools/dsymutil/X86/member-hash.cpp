// Check that dsymutil's accelerator table hashing walks DW_AT_specification to
// build a member's fully qualified name. The typedef below is nested in a
// member function whose definition only reaches the enclosing struct through
// DW_AT_specification.
//
// Compile with:
//   clang -g -c member-hash.cpp -o member-hash/2.o

// RUN: dsymutil --linker classic -oso-prepend-path %p/../Inputs/member-hash \
// RUN:   -y -f %p/../Inputs/member-hash/debug-map.map -o - \
// RUN:   | llvm-dwarfdump -apple-types - | FileCheck %s

// RUN: dsymutil --linker parallel -oso-prepend-path %p/../Inputs/member-hash \
// RUN:   -y -f %p/../Inputs/member-hash/debug-map.map -o - \
// RUN:   | llvm-dwarfdump -apple-types - | FileCheck %s

struct S {
  int foo();
};

int S::foo() {
  typedef int T;
  return (T)42;
}

void foo() {
  S s;
  s.foo();
}

// The DIE and string offsets differ between the two linkers; the hash in
// Atom[3] is what this test pins down.

// CHECK: String: 0x{{[0-9a-f]+}} "T"
// CHECK-NEXT: Data 0 [
// CHECK-NEXT:  Atom[0]: 0x{{[0-9a-f]+}}
// CHECK-NEXT:  Atom[1]: 0x0016
// CHECK-NEXT:  Atom[2]: 0x00
// CHECK-NEXT:  Atom[3]: 0xa415d958
// CHECK-NEXT:]
