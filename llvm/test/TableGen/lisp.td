// RUN: llvm-tblgen %s

// CHECK:      def One {
// CHECK-NEXT:   list<string> names = ["Jeffrey Sinclair"];
// CHECK-NEXT:   string element = "Jeffrey Sinclair";
// CHECK-NEXT:   list<string> rest = [];
// CHECK-NEXT:   int null = 1;
// CHECK-NEXT: }
// CHECK-NEXT: def Three {
// CHECK-NEXT:   list<string> names = ["Tom", "Dick", "Harry"];
// CHECK-NEXT:   string element = "Tom";
// CHECK-NEXT:   list<string> rest = ["Dick", "Harry"];
// CHECK-NEXT:   int null = 0;
// CHECK-NEXT: }

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
