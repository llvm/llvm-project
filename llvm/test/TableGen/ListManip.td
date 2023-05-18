// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class Bli<string _t>
{
  string t = _t;
}

class Bla<list<Bli> _bli>
: Bli<!head(_bli).t>
{
}
