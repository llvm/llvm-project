// UNSUPPORTED: system-windows
//
// Test to verify we are corectly generating anonymous flags when parsing
// anonymous class and unnamed structs from DWARF to the a clang AST node.

// RUN: %clang++ -g -c -o %t.o %s
// RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck %s

struct A {
  struct {
    int x;
  };
  struct {
    int y;
  } C;
} a;

// CHECK: A::(anonymous struct)
// CHECK: |-DefinitionData is_anonymous pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK: A::(anonymous struct)
// CHECK: |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
