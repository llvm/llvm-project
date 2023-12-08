// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Base<list<int> v> {
  list<int> values = v;
}

class Derived<list<int> v> : Base<!if(!empty(v),[0],v)>;

multiclass Multi<list<list<list<int>>> v> {
  def ONE : Derived<!if(!empty(v),[]<int>,!if(!empty(!head(v)),[]<int>,v[0][0]))>;
  def TWO : Derived<!if(!empty(v),[]<int>,!if(!empty(!tail(v)),!if(!empty(!head(v)),[]<int>,v[0][0]),v[1][0]))>;
}

defm Def : Multi<[[[]],[[1, 2, 3]]]>;

// CHECK: values = [0]
// CHECK: values = [1, 2, 3]
