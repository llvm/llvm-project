// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK-NOT: zing = 4

class C1<int A, string B> { 
  int bar = A;
  string thestr = B;
  int zing;
}

def T : C1<4, "blah">;

multiclass t<int a> {
  def S1 : C1<a, "foo"> {
    int foo = 4;
    let bar = 1;
  }
  def S2 : C1<a, "bar">;
}

multiclass s<int a> {
  def S3 : C1<a, "moo"> {
    int moo = 3;
    let bar = 1;
  }
  def S4 : C1<a, "baz">;
}

defm FOO : t<42>, s<24>;

def T4 : C1<6, "foo">;

let zing = 4 in
  defm BAZ : t<3>, s<4>;
