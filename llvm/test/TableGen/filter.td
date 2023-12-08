// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

defvar EmptyList = []<string>;
defvar OneList = ["foo"];
defvar WordList = ["foo", "bar", "zoo", "foo", "snork", "snork", "quux"];

class Predicate;
def pred1 : Predicate;
def pred2 : Predicate;
def pred3 : Predicate;
def pred4 : Predicate;
def pred5 : Predicate;

class DeduplicatePredList<list<Predicate> predlist, Predicate pred> {
  list<Predicate> ret =
      !listconcat([pred], !filter(item, predlist,
                                  !ne(!cast<string>(item), !cast<string>(pred))));
}

// CHECK: def rec1
// CHECK:   list1 = [];
// CHECK:   list2 = [];
// CHECK:   list3 = ["foo"];
// CHECK:   list4 = [];
// CHECK:   list5 = ["foo", "bar", "zoo", "foo", "snork", "snork", "quux"];
// CHECK:   list6 = [];

def rec1 {
  list<string> list1 = !filter(item, EmptyList, true);
  list<string> list2 = !filter(item, EmptyList, false);
  list<string> list3 = !filter(item, OneList, true);
  list<string> list4 = !filter(item, OneList, false);
  list<string> list5 = !filter(item, WordList, true);
  list<string> list6 = !filter(item, WordList, false);
}

// CHECK: def rec2
// CHECK:   list1 = ["foo", "foo"];
// CHECK:   list2 = ["bar", "zoo", "snork", "snork", "quux"];
// CHECK:   list3 = ["snork", "snork", "quux"];

def rec2 {
  list<string> list1 = !filter(item, WordList, !eq(item, "foo"));
  list<string> list2 = !filter(item, WordList, !ne(item, "foo"));
  list<string> list3 = !filter(item, WordList, !ge(!size(item), 4));
}

// CHECK: def rec3
// CHECK:   list1 = [4, 5, 6, 7, 8, 9, 10];
// CHECK:   list2 = [4, 5, 6, 7, 8];

def rec3 {
  list<int> list1 = !filter(num, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], !gt(num, 3));
  list<int> list2 = !filter(num, list1, !lt(num, 9));
}

// CHECK: def rec4
// CHECK:   duplist = [pred1, pred2, pred1, pred3, pred4, pred1, pred5];
// CHECK:   deduplist = [pred1, pred2, pred3, pred4, pred5];

def rec4 {
  list<Predicate> duplist = [pred1, pred2, pred1, pred3, pred4, pred1, pred5];
  list<Predicate> deduplist = DeduplicatePredList<duplist, pred1>.ret;  
}

#ifdef ERROR1

// ERROR1: could not be fully resolved

def rec9 {
  list<string> list1 = !filter(item, WordList, !if(true, "oops!", "wrong!"));
}

#endif
