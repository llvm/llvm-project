// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

// CHECK: error: adding record of incompatible type 'A' to defset

class A<int a> {
  int Num = a;
}

class B<int a> : A<a>;

defset list<B> Bs = {
  def A0 : A<1>;
}
