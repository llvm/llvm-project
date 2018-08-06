// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: Classes
// CHECK: Sr
// CHECK: Jr
// CHECK: "NAME"

// CHECK: Defs

// CHECK: def DX {
// CHECK:   list<string> x = ["0", "1", "2"];
// CHECK: }

// CHECK: def DY {
// CHECK:   list<int> y = [5, 7];
// CHECK: }

// CHECK: Jr
// CHECK: Sr

class A<list<string> names> {
  list<string> Names = names;
}

class B<list<string> names> : A<!foreach(name, names, !strconcat(name, ", Sr."))>;

class C<list<string> names> : A<!foreach(name, names, !strconcat(name, ", Jr."))>;

class D<list<string> names> : A<!foreach(name, names, !subst("NAME", "John Smith", name))>;

class Names {
  list<string> values = ["Ken Griffey", "Seymour Cray"];
}

def People : Names;

def Seniors : B<People.values>;
def Juniors : C<People.values>;
def Smiths : D<["NAME", "Jane Smith"]>;
def Unprocessed : D<People.values>;

class X<list<int> a> {
    list<string> x = !foreach(num, a, !cast<string>(num));
}

def DX : X<[0, 1, 2]>;

class Y<list<int> a> {
  list<int> y = !foreach(num, a, !add(!add(4, num), !add(1, num)));
}

def DY: Y<[0, 1]>;
