// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// Tests for the true and false literals.

defvar otherwise = true;
defvar do_it = true;

// CHECK: def rec1
// CHECK:   bit flag1 = 1;
// CHECK:   bit flag2 = 0;
// CHECK:   int true_int = 1;

def rec1 {
  bit flag1 = true;
  bit flag2 = false;
  int true_int = true;
}

// CHECK: def rec2_true

if true then
  def rec2_true {}
else
  def rec2_bad {}

// CHECK: def rec3_false

if false then
  def rec3_bad {}
else
  def rec3_false {}

// CHECK: def rec4
// CHECK:   int value = 52;

def rec4 {
  int value = !add(10, !if(!and(do_it, true), 42, 0));
}

// CHECK: def rec5
// CHECK:    string name = "snork";

def rec5 {
  string name = !cond(false:       "foo",
                      !not(do_it): "bar",
                      otherwise:   "snork");
}

// CHECK: def rec6
// CHECK:   bit xorFF = 0;
// CHECK:   bit xorFT = 1;
// CHECK:   bit xorTF = 1;
// CHECK:   bit xorTT = 0;

def rec6 {
  bit xorFF = !xor(false, false);
  bit xorFT = !xor(false, true);
  bit xorTF = !xor(true, false);
  bit xorTT = !xor(true, true);
}

// CHECK: def rec7
// CHECK:   bits<3> flags = { 1, 0, 1 };

def rec7 {
  bits<3> flags = { true, false, true };
}

// `!and` and `!or` should be short-circuited such that any of the `!head` or
// `!tail` on empty list below will never be evaluated.
// CHECK: def rec8
// CHECK:   bit v = 0;
// CHECK:   int v2 = -1;
// CHECK:   list<int> newSeq = [];
// CHECK:   list<int> newSeq2 = [];

class Foo <list<int> seq = []> {
  bit v = !and(false, !head(seq));
  int v2 = !or(-1, !head(seq));

  bit unresolved = !ne(!find(NAME, "BAR"), -1);
  list<int> newSeq  = !if(!and(false, unresolved), !tail(seq), seq);
  list<int> newSeq2 = !if(!or(-1, unresolved), seq, !tail(seq));
}

def rec8 : Foo<>;

#ifdef ERROR1
// ERROR1: Record name '1' is not a string

def true {}
#endif

