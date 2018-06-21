// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: Classes
// CHECK: Sr
// CHECK: Jr
// CHECK: "NAME"

// CHECK: Defs
// CHECK: Jr
// CHECK: Sr

// Variables for foreach
class decls {
  string name;
}

def Decls : decls;

class A<list<string> names> {
  list<string> Names = names;
}

class B<list<string> names> : A<!foreach(Decls.name, names, !strconcat(Decls.name, ", Sr."))>;

class C<list<string> names> : A<!foreach(Decls.name, names, !strconcat(Decls.name, ", Jr."))>;

class D<list<string> names> : A<!foreach(Decls.name, names, !subst("NAME", "John Smith", Decls.name))>;

class Names {
  list<string> values = ["Ken Griffey", "Seymour Cray"];
}

def People : Names;

def Seniors : B<People.values>;
def Juniors : C<People.values>;
def Smiths : D<["NAME", "Jane Smith"]>;
def Unprocessed : D<People.values>;
