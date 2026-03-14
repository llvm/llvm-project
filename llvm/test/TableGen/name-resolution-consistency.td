// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: def B0a {
// CHECK:   string e = "B0";
// CHECK: }

// CHECK: def B0ba {
// CHECK:   string a = "B0b";
// CHECK:   string b = "B0";
// CHECK: }

// CHECK: def B0bys {
// CHECK:   string f = "B0b";
// CHECK:   string g = "B0";
// CHECK: }

// CHECK: def B0cza {
// CHECK:   string a = "B0cz";
// CHECK:   string b = "B0";
// CHECK: }

// CHECK: def B0czyt {
// CHECK:   string f = "B0cz";
// CHECK:   string g = "B0";
// CHECK: }

// CHECK: def C0 {
// CHECK:   string a = "C0";
// CHECK:   string b = "C0";
// CHECK:   string c = "a";
// CHECK: }

// CHECK: def D0a {
// CHECK:   string a = "D0a";
// CHECK:   string b = "D0a";
// CHECK:   string c = "D0";
// CHECK: }

// CHECK: def D0b {
// CHECK:   string a = "D0b";
// CHECK:   string b = "D0b";
// CHECK:   string c = "a";
// CHECK: }

// CHECK: def xB0b {
// CHECK:   string c = "B0b";
// CHECK:   string d = "B0";
// CHECK: }

// CHECK: def xB0cz {
// CHECK:   string c = "B0cz";
// CHECK:   string d = "B0";
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

class Cbase {
  string a = NAME;
}

class C<string arg> : Cbase {
  string b = NAME;
  string c = arg;
}

def C0 : C<"a">;

multiclass D<string arg> {
  def a : C<NAME>;
  def b : C<arg>;
}

defm D0 : D<"a">;
