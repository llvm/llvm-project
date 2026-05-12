// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s

defvar EmptyStrings = []<string>;
defvar EmptyInts = []<int>;
defvar Creatures = ["cephalopod", "axolotl", "dragonfly", "barracuda"];
defvar Nums = [5, 2, 8, 1, 3];

// CHECK: def rec1
// CHECK:   sorted_strings = ["axolotl", "barracuda", "cephalopod", "dragonfly"];
// CHECK:   sorted_ints = [1, 2, 3, 5, 8];
// CHECK:   sorted_empty_strings = [];
// CHECK:   sorted_empty_ints = [];

def rec1 {
  list<string> sorted_strings = !sort(item, Creatures, item);
  list<int> sorted_ints = !sort(n, Nums, n);
  list<string> sorted_empty_strings = !sort(item, EmptyStrings, item);
  list<int> sorted_empty_ints = !sort(n, EmptyInts, n);
}

// Stable sort: equal keys preserve original relative order.
// CHECK: def rec2
// CHECK:   stable = ["a", "a", "b", "b", "c"];

defvar DuplicateWords = ["b", "a", "b", "c", "a"];
def rec2 {
  list<string> stable = !sort(item, DuplicateWords, item);
}

// Sort records by a string field or an int field.
class Thing<string N, int V> {
  string Name = N;
  int Value = V;
}
def thing_c : Thing<"charlie", 30>;
def thing_a : Thing<"alpha",   10>;
def thing_b : Thing<"beta",    20>;

defvar Things = [thing_c, thing_a, thing_b];

// CHECK: def rec3
// CHECK:   by_name  = [thing_a, thing_b, thing_c];
// CHECK:   by_value = [thing_a, thing_b, thing_c];

def rec3 {
  list<Thing> by_name  = !sort(t, Things, t.Name);
  list<Thing> by_value = !sort(t, Things, t.Value);
}

// Sort an already-sorted list — should be a no-op.
// CHECK: def rec4
// CHECK:   already = ["a", "b", "c"];

def rec4 {
  list<string> already = !sort(item, ["a", "b", "c"], item);
}

// Sort a single-element list.
// CHECK: def rec5
// CHECK:   one = ["only"];

def rec5 {
  list<string> one = !sort(item, ["only"], item);
}

#ifdef ERROR1
// Dag is not a valid second argument.
// ERROR1: !sort must have a list argument
def myop;
defvar mydag = (myop);
def err1 {
  list<string> bad = !sort(x, mydag, !cast<string>(x));
}
#endif

#ifdef ERROR2
// Key that cannot be resolved to int or string leaves the op unfolded.
// ERROR2: could not be fully resolved
def err2 {
  list<Thing> bad = !sort(t, Things, t);
}
#endif
