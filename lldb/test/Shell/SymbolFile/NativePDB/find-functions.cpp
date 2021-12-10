// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /GR- /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb

// RUN: lldb-test symbols --find=function --name=main --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-MAIN

// RUN: lldb-test symbols --find=function --name=static_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC

// RUN: lldb-test symbols --find=function --name=varargs_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VAR

// RUN: lldb-test symbols --find=function --name=Struct::simple_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-SIMPLE

// RUN: lldb-test symbols --find=function --name=Struct::virtual_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VIRTUAL

// RUN: lldb-test symbols --find=function --name=Struct::static_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC-METHOD

// RUN: lldb-test symbols --find=function --name=Struct::overloaded_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-OVERLOAD

struct Struct {
  int simple_method() {
    return 1;
  }

  virtual int virtual_method() {
    return 2;
  }

  static int static_method() {
    return 3;
  }

  int overloaded_method() {
    return 4 + overloaded_method('a') + overloaded_method('a', 1);
  }
protected:
  virtual int overloaded_method(char c) {
    return 5;
  }
private:
  static int overloaded_method(char c, int i, ...) {
    return 6;
  }
};

Struct s;

static int static_fn() {
  return 42;
}

int varargs_fn(int x, int y, ...) {
  return x + y;
}

int main(int argc, char **argv) {
  return static_fn() + varargs_fn(argc, argc) + s.simple_method() +
  Struct::static_method() + s.virtual_method() + s.overloaded_method();
}

// FIND-MAIN:      Function: id = {{.*}}, name = "main"
// FIND-MAIN-NEXT: FuncType: id = {{.*}}, compiler_type = "int (int, char **)"

// FIND-STATIC:      Function: id = {{.*}}, name = "{{.*}}static_fn{{.*}}"
// FIND-STATIC-NEXT: FuncType: id = {{.*}}, compiler_type = "int (void)"

// FIND-VAR:      Function: id = {{.*}}, name = "{{.*}}varargs_fn{{.*}}"
// FIND-VAR-NEXT: FuncType: id = {{.*}}, compiler_type = "int (int, int, ...)"

// FIND-SIMPLE:      Function: id = {{.*}}, name = "{{.*}}Struct::simple_method{{.*}}"
// FIND-SIMPLE-NEXT: FuncType: id = {{.*}}, compiler_type = "int (void)"

// FIND-VIRTUAL:      Function: id = {{.*}}, name = "{{.*}}Struct::virtual_method{{.*}}"
// FIND-VIRTUAL-NEXT: FuncType: id = {{.*}}, compiler_type = "int (void)"

// FIND-STATIC-METHOD:      Function: id = {{.*}}, name = "{{.*}}Struct::static_method{{.*}}"
// FIND-STATIC-METHOD-NEXT: FuncType: id = {{.*}}, compiler_type = "int (void)"

// FIND-OVERLOAD: Function: id = {{.*}}, name = "{{.*}}Struct::overloaded_method{{.*}}"
// FIND-OVERLOAD: FuncType: id = {{.*}}, compiler_type = "int (void)"
// FIND-OVERLOAD: FuncType: id = {{.*}}, compiler_type = "int (char)"
// FIND-OVERLOAD: FuncType: id = {{.*}}, compiler_type = "int (char, int, ...)"
