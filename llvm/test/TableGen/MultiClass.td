// RUN: llvm-tblgen %s | grep "zing = 4" | count 2
// XFAIL: vg_leak

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

defm FOO : t<42>;

def T4 : C1<6, "foo">;

let zing = 4 in
  defm BAZ : t<3>;

