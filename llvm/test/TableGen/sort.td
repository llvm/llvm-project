// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR_NONLIST %s 2>&1 | FileCheck --check-prefix=ERROR_NONLIST %s
// RUN: not llvm-tblgen -DERROR_KEYTYPE %s 2>&1 | FileCheck --check-prefix=ERROR_KEYTYPE %s

// Sort an already-sorted list — should be a no-op.
// CHECK-LABEL: def idempotent
// CHECK:   already = ["a", "b", "c"];
def idempotent {
  list<string> already = !sort(item, ["a", "b", "c"], item);
}

// Sort records by a string field or an int field. The sort is stable, so order
// is preserved on elements with equal key.
// CHECK-LABEL: def key_expr
// CHECK:   by_name  = [thing_a20, thing_a10, thing_b20, thing_c30, thing_d10];
// CHECK:   by_value = [thing_a10, thing_d10, thing_a20, thing_b20, thing_c30];
class Thing<string N, int V> {
  string Name = N;
  int Value = V;
}
def thing_a10 : Thing<"alpha",   10>;
def thing_a20 : Thing<"alpha",   20>;
def thing_b20 : Thing<"beta",    20>;
def thing_c30 : Thing<"charlie", 30>;
def thing_d10 : Thing<"delta",   10>;
defvar Things = [thing_c30, thing_a20, thing_a10, thing_b20, thing_d10];
def key_expr {
  list<Thing> by_name  = !sort(t, Things, t.Name);
  list<Thing> by_value = !sort(t, Things, t.Value);
}

// Sort a single-element list.
// CHECK-LABEL: def single
// CHECK:   one = ["only"];
def single {
  list<string> one = !sort(item, ["only"], item);
}

// CHECK-LABEL: def sorted
// CHECK:   sorted_strings = ["axolotl", "barracuda", "cephalopod", "dragonfly"];
// CHECK:   sorted_ints = [1, 2, 3, 5, 8];
// CHECK:   sorted_empty_strings = [];
// CHECK:   sorted_empty_ints = [];
defvar EmptyStrings = []<string>;
defvar Creatures = ["cephalopod", "axolotl", "dragonfly", "barracuda"];
defvar EmptyInts = []<int>;
defvar Nums = [5, 2, 8, 1, 3];
def sorted {
  list<string> sorted_strings = !sort(item, Creatures, item);
  list<int> sorted_ints = !sort(n, Nums, n);
  list<string> sorted_empty_strings = !sort(item, EmptyStrings, item);
  list<int> sorted_empty_ints = !sort(n, EmptyInts, n);
}

#ifdef ERROR_NONLIST
// Dag is not a valid second argument.
def myop;
defvar mydag = (myop);
def err_nonlist {
  // ERROR_NONLIST: sort.td:[[@LINE+1]]:38: error: !sort must have a list argument
  list<string> bad = !sort(x, mydag, !cast<string>(x));
}
#endif

#ifdef ERROR_KEYTYPE
// Key that cannot be resolved to int or string leaves the op unfolded.
// ERROR_KEYTYPE: sort.td:[[@LINE+1]]:5: error: Initializer of 'bad' in 'err_keytype' could not be fully resolved
def err_keytype {
  list<Thing> bad = !sort(t, Things, t);
}
#endif
