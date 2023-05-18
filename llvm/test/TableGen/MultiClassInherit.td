// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// "zing = 4" x 28
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
// CHECK: zing = 4
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

multiclass t1<int a1> {
  def S1 : C1<a1, "foo"> {
    int foo = 4;
    let bar = 1;
  }
  def S2 : C1<a1, "bar">;
}

multiclass t2<int a2> {
  def S3 : C1<a2, "foo"> {
    int foo = 4;
    let bar = 1;
  }
  def S4 : C1<a2, "bar">;
}

multiclass s1<int as1, int bs1> : t1<as1> {
  def S5 : C1<bs1, "moo"> {
    int moo = 3;
    let bar = 1;
  }
  def S6 : C1<bs1, "baz">;
}

multiclass s2<int as2> : t1<as2>, t2<as2>;

multiclass s3<int as3, int bs3> : t1<as3>, t2<as3> {
  def S7 : C1<bs3, "moo"> {
    int moo = 3;
    let bar = 1;
  }
  def S8 : C1<bs3, "baz">;
}

let zing = 4 in
defm FOO1 : s1<42, 24>;

let zing = 4 in
defm FOO2 : s2<99>;

let zing = 4 in
defm FOO3 : s3<84, 48>;

def T4 : C1<6, "foo">;

let zing = 4 in
  defm BAZ1 : s1<3, 4>;

let zing = 4 in
  defm BAZ2 : s2<5>;

let zing = 4 in
  defm BAZ3 : s3<6, 7>;

