// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /GR- /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb

// RUN: lldb-test symbols --find=function --name=main --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-MAIN
// RUN: lldb-test symbols --find=function --name=main --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-NO-FUNCTION
// RUN: lldb-test symbols --find=function --name=main --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-MAIN

// RUN: lldb-test symbols --find=function --name=static_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC
// RUN: lldb-test symbols --find=function --name=static_fn --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-NO-FUNCTION
// RUN: lldb-test symbols --find=function --name=static_fn --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC

// RUN: lldb-test symbols --find=function --name=varargs_fn --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VAR
// RUN: lldb-test symbols --find=function --name=varargs_fn --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-NO-FUNCTION
// RUN: lldb-test symbols --find=function --name=varargs_fn --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VAR

// RUN: lldb-test symbols --find=function --name=Struct::simple_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-SIMPLE
// RUN: lldb-test symbols --find=function --name=Struct::simple_method --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-SIMPLE
// RUN: lldb-test symbols --find=function --name=Struct::simple_method --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-SIMPLE

// RUN: lldb-test symbols --find=function --name=Struct::virtual_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VIRTUAL
// RUN: lldb-test symbols --find=function --name=Struct::virtual_method --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VIRTUAL
// RUN: lldb-test symbols --find=function --name=Struct::virtual_method --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-VIRTUAL

// RUN: lldb-test symbols --find=function --name=Struct::static_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC-METHOD
// RUN: lldb-test symbols --find=function --name=Struct::static_method --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-NO-FUNCTION
// RUN: lldb-test symbols --find=function --name=Struct::static_method --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-STATIC-METHOD

// RUN: lldb-test symbols --find=function --name=Struct::overloaded_method --function-flags=full %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-OVERLOAD-FULL
// RUN: lldb-test symbols --find=function --name=Struct::overloaded_method --function-flags=method %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-OVERLOAD-METHOD
// RUN: lldb-test symbols --find=function --name=Struct::overloaded_method --function-flags=base %t.exe \
// RUN:     | FileCheck %s --check-prefix=FIND-OVERLOAD-BASE

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

class Class {
public:
  bool overloaded_method() {
    return false;
  }
  bool overloaded_method(int i) {
    return i > 0;
  }
  static int overloaded_method(bool b) {
    return b ? 1 : 2;
  }
};

char overloaded_method() {
  return 0;
}
char overloaded_method(int i) {
  return 0;
}

Struct s;
Class c;

static int static_fn() {
  return 42;
}

int varargs_fn(int x, int y, ...) {
  return x + y;
}

int main(int argc, char **argv) {
  return static_fn() + varargs_fn(argc, argc) + s.simple_method() +
  Struct::static_method() + s.virtual_method() + s.overloaded_method() + 
  Class::overloaded_method(false) + c.overloaded_method(1) + c.overloaded_method() 
  + overloaded_method() + overloaded_method(1);
}

// FIND-MAIN:      Function: id = {{.*}}, name = "main"
// FIND-MAIN-NEXT: FuncType: id = {{.*}}, compiler_type = "int (int, char **)"

// FIND-NO-FUNCTION: Found 0 functions

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

// FIND-OVERLOAD-FULL-NOT: "Class::overloaded_method"
// FIND-OVERLOAD-FULL-NOT: "overloaded_method"
// FIND-OVERLOAD-FULL: Function: id = {{.*}}, name = "{{.*}}Struct::overloaded_method{{.*}}"
// FIND-OVERLOAD-FULL: FuncType: id = {{.*}}, compiler_type = "int (void)"
// FIND-OVERLOAD-FULL: FuncType: id = {{.*}}, compiler_type = "int (char)"
// FIND-OVERLOAD-FULL: FuncType: id = {{.*}}, compiler_type = "int (char, int, ...)"

// FIND-OVERLOAD-BASE-DAG: Function: id = {{.*}}, name = "{{.*}}Struct::overloaded_method{{.*}}"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "int (void)"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "int (char)"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "int (char, int, ...)"
// FIND-OVERLOAD-BASE-DAG: Function: id = {{.*}}, name = "Class::overloaded_method"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "_Bool (void)"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "_Bool (int)"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "int (_Bool)"
// FIND-OVERLOAD-BASE-DAG: Function: id = {{.*}}, name = "overloaded_method"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "char (void)"
// FIND-OVERLOAD-BASE-DAG: FuncType: id = {{.*}}, compiler_type = "char (int)"

// FIND-OVERLOAD-METHOD-NOT: "overloaded_method"
// FIND-OVERLOAD-METHOD-DAG: Function: id = {{.*}}, name = "{{.*}}Struct::overloaded_method{{.*}}"
// FIND-OVERLOAD-METHOD-DAG: FuncType: id = {{.*}}, compiler_type = "int (void)"
// FIND-OVERLOAD-METHOD-DAG: FuncType: id = {{.*}}, compiler_type = "int (char)"
// FIND-OVERLOAD-METHOD-DAG: Function: id = {{.*}}, name = "Class::overloaded_method"
// FIND-OVERLOAD-METHOD-DAG: FuncType: id = {{.*}}, compiler_type = "_Bool (void)"
// FIND-OVERLOAD-METHOD-DAG: FuncType: id = {{.*}}, compiler_type = "_Bool (int)"
