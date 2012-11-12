// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK: No subst
// CHECK: No foo
// CHECK: RECURSE foo

class Recurse<string t> {
  string Text = t;
}

class Text<string text> : 
  Recurse<!subst("RECURSE", "RECURSE", !subst("NORECURSE", "foo", text))>;

def Ok1 : Text<"No subst">;
def Ok2 : Text<"No NORECURSE">;
def Trouble : Text<"RECURSE NORECURSE">;
