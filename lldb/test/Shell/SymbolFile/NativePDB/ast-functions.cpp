// clang-format off
// REQUIRES: lld

// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s

// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-functions.lldbinit 2>&1 | FileCheck %s

static int static_fn() {
  return 42;
}

int varargs_fn(int x, int y, ...) {
  return x + y;
}

int main(int argc, char **argv) {
  return static_fn() + varargs_fn(argc, argc);
}

// CHECK:      TranslationUnitDecl
// CHECK-NEXT: |-FunctionDecl {{.*}} main 'int (int, char **)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} argc 'int'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} argv 'char **'
// CHECK-NEXT: |-FunctionDecl {{.*}} static_fn 'int ()' static
// CHECK-NEXT: `-FunctionDecl {{.*}} varargs_fn 'int (int, int, ...)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} x 'int'
// CHECK-NEXT:   `-ParmVarDecl {{.*}} y 'int'
