// RUN: llvm-tblgen %s | grep ""
// XFAIL: vg_leak

class List<list<string> n> {
  list<string> names = n;
}

class CAR<string e> {
  string element = e;
}

class CDR<list<string> r, int n> {
  list<string> rest = r;
  int null = n;
}

class NameList<list<string> Names> :
  List<Names>, CAR<!head(Names)>, CDR<!tail(Names), !empty(!tail(Names))>;

def Three : NameList<["Tom", "Dick", "Harry"]>;

def One : NameList<["Jeffrey Sinclair"]>;
