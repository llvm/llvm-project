// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class Bla<string t>
{
  string blu = t;
}

class Bli<Bla t>
{
  Bla bla = t;
}

def a : Bli<Bla<"">>;
def b : Bla<!cast<Bla>(a.bla).blu>; // works
def c : Bla<a.bla.blu>; // doesn't work: Cannot access field 'blu' of value 'a.bla'
