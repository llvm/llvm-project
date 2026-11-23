// RUN: llvm-tblgen %s

class Bli<string _t>
{
  string t = _t;
}

class Bla<list<Bli> _bli>
: Bli<!head(_bli).t>
{
}
