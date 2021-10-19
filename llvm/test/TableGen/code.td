// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// XFAIL: vg_leak

// CHECK: def A1
// CHECK:   code CodeCode = [{code here;}]
// CHECK:   code StringCode = [{code here;}]

// CHECK: def A2
// CHECK:   string CodeCode = "string here"
// CHECK:   string StringCode = "string here"

// CHECK: def B1
// CHECK:   string CodeCode = "with paste 7"
// CHECK:   string StringCode = "with paste 7"

// CHECK: def C1
// CHECK:   code CodeCode = [{with concat 42}]
// CHECK:   code StringCode = [{with concat 42}]

// CHECK: def D1
// CHECK:   code CodeCode = [{with concat 108!}]
// CHECK:   code StringCode = [{with concat 108!}]

class A<code c> {
  code CodeCode = c;
  string StringCode = c;
}

def A1 : A<[{code here;}]>;
def A2 : A<"string here">;

class B<int i> : A<"with paste " # i>;
class C<int i> : A<!strconcat([{with concat }], !cast<string>(i))>;
class D<int i> : A<!strconcat([{with concat }], !cast<string>(i), "!")>;

def B1 : B<7>;
def C1 : C<42>;
def D1 : D<108>;

#ifdef ERROR1

// ERROR1: the 'code' type is not allowed

def Zerror1 {
  code Code = !cast<code>("i = 0;");
}

#endif
