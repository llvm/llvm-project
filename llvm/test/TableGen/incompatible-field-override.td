// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// Redeclaring a field with an incompatible type used to leave the field's
// Value as nullptr, causing an isa<UnsetInit>(nullptr) assert when a later
// def tried to resolve an expression referring to that field.

class X {
  int F = 0;
  int D = !add(F, 1);
}

// CHECK: error: Field 'F' of type 'int' is incompatible with value '"0"' of type 'string'
// CHECK-NOT: Assertion `Val && "isa<> used on a null pointer"'
class Y : X  {
  string F = "0";
}

def y : Y;
