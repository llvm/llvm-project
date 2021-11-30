// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb

// RUN: lldb-test symbols --find=function --name=main --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-MAIN

// RUN: lldb-test symbols --find=function --name=static_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC

// RUN: lldb-test symbols --find=function --name=varargs_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VAR

static int static_fn() {
  return 42;
}

int varargs_fn(int x, int y, ...) {
  return x + y;
}

int main(int argc, char **argv) {
  return static_fn() + varargs_fn(argc, argc);
}

// FIND-MAIN:      Function: id = {{.*}}, name = "main"
// FIND-MAIN-NEXT: FuncType: id = {{.*}}, compiler_type = "int (int, char **)"

// FIND-STATIC:      Function: id = {{.*}}, name = "static_fn"
// FIND-STATIC-NEXT: FuncType: id = {{.*}}, compiler_type = "int (void)"

// FIND-VAR:      Function: id = {{.*}}, name = "varargs_fn"
// FIND-VAR-NEXT: FuncType: id = {{.*}}, compiler_type = "int (int, int, ...)"
