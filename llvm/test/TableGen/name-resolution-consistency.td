// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// This test demonstrates a number of inconsistencies in how NAME is resolved
// and record names are constructed.
//
// The TODO lines describe a suggested consistent behavior that would result
// from:
//  (1) Treating NAME as an implicit multiclass template argument and
//  (2) always storing the name of (non-anonymous) prototype records in
//      multiclasses with at least one explicit reference to NAME.
//
// Unfortunately, several backends (including X86) rely quite heavily on the
// current inconsistent behavior and would have to be fixed.

// CHECK: def B0a {
// CHECK:   string e = "B0";
// CHECK: }

// CHECK: def B0ba {
// TODO: expect "B0b" here
// CHECK:   string a = "B0";
// CHECK:   string b = "B0";
// CHECK: }

// CHECK: def B0cza {
// TODO: expect "B0cz" here
// CHECK:   string a = "B0";
// CHECK:   string b = "B0";
// CHECK: }

// TODO: expect this to be named 'xB0b'
// CHECK: def B0xb {
// TODO: expect "B0b" here
// CHECK:   string c = "b";
// CHECK:   string d = "b";
// CHECK: }

// TODO: expect this to be named B0bys
// CHECK: def B0ys {
// TODO: expect "B0b" here
// CHECK:   string f = "b";
// CHECK:   string g = "b";
// CHECK: }

// CHECK: def xB0cz {
// CHECK:   string c = "B0cz";
// CHECK:   string d = "B0cz";
// CHECK: }

// TODO: expect this to be named B0czyt
// CHECK: def yt {
// CHECK:   string f = "B0cz";
// CHECK:   string g = "B0cz";
// CHECK: }

multiclass A<string p, string q> {
  def a {
    string a = NAME;
    string b = p;
  }

  def x # NAME {
    string c = NAME;
    string d = p;
  }

  def y # q {
    string f = NAME;
    string g = p;
  }
}

multiclass B<string name, string t> {
  def a {
    string e = NAME;
  }

  defm b : A<NAME, "s">;

  defm NAME # c # name : A<NAME, t>;
}

defm B0 : B<"z", "t">;
