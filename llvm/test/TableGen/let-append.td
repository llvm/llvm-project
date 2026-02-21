// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Test 'let append' and 'let prepend' syntax.

def op;

class Base {
  list<int> items = [1, 2];
  string text = "hello";
  dag d = (op);
}

class WithCode {
  code body = [{ int x = 0; }];
}

class WithUnset {
  list<int> vals = ?;
  string msg = ?;
}

// Multi-level inheritance accumulation.
class Middle : Base {
  let append items = [3];
  let append text = " world";
}

// Multiple inheritance classes.
class BaseA {
  list<int> items = [1, 2];
  string text = "a";
}

class BaseB {
  list<int> items = [3, 4];
  string text = "b";
}

// Diamond inheritance classes.
class DiamondBase {
  list<int> items = [1];
}

class Left : DiamondBase {
  let append items = [2];  // items = [1, 2]
}

class Right : DiamondBase {
  let append items = [3];  // items = [1, 3]
}

// Template argument class.
class Parameterized<list<int> init> {
  list<int> items = init;
}

// Multiclass with append/prepend in body.
multiclass MC<list<int> extra> {
  def _a : Base {
    let append items = extra;
  }
  def _b : Base {
    let prepend items = extra;
  }
}

// Test that 'append' and 'prepend' can be used as field names
// (contextual keywords, not reserved).
class HasAppendField {
  list<int> append = [1, 2];
  list<int> prepend = [3, 4];
}

// --- Definitions (CHECK lines in alphabetical order of def names) ---

// CHECK: def AppendCode
// CHECK: code body = [{ int x = 0;  int y = 1; }]
def AppendCode : WithCode {
  let append body = [{ int y = 1; }];
}

// CHECK: def AppendDag
// CHECK: dag d = (op 3:$a);
def AppendDag : Base {
  let append d = (op 3:$a);
}

// CHECK: def AppendList
// CHECK: list<int> items = [1, 2, 3, 4];
def AppendList : Base {
  let append items = [3, 4];
}

// CHECK: def AppendString
// CHECK: string text = "hello world";
def AppendString : Base {
  let append text = " world";
}

// CHECK: def AppendUnset
// CHECK: list<int> vals = [1];
// CHECK: string msg = "hi";
def AppendUnset : WithUnset {
  let append vals = [1];
  let append msg = "hi";
}

// Test sequential append + prepend on the same field.
// CHECK: def Both
// CHECK: list<int> items = [0, 1, 2, 3];
def Both : Base {
  let append items = [3];
  let prepend items = [0];
}

// CHECK: def ContextualKeyword
// CHECK: list<int> append = [1, 2, 5];
// CHECK: list<int> prepend = [0, 3, 4];
def ContextualKeyword : HasAppendField {
  let append append = [5];
  let prepend prepend = [0];
}

// Test diamond inheritance: Right is the last parent, so only Right's
// accumulated value survives. Left's append is lost.
// CHECK: def Diamond
// CHECK: list<int> items = [1, 3, 4];
def Diamond : Left, Right {
  let append items = [4];
}

// Test let append on a field set by a template argument.
// CHECK: def FromTemplateArg
// CHECK: list<int> items = [1, 2, 3];
def FromTemplateArg : Parameterized<[1, 2]> {
  let append items = [3];
}

// Test let append in multiclass body with defm.
// CHECK: def MCTest_a
// CHECK: list<int> items = [1, 2, 10, 20];
// CHECK: def MCTest_b
// CHECK: list<int> items = [10, 20, 1, 2];
defm MCTest : MC<[10, 20]>;

// Test multiple inheritance: last parent class wins, then append applies.
// CHECK: def MultiInherit
// CHECK: list<int> items = [3, 4, 5];
// CHECK: string text = "b!";
def MultiInherit : BaseA, BaseB {
  let append items = [5];
  let append text = "!";
}

// CHECK: def MultiLevel
// CHECK: list<int> items = [1, 2, 3, 4];
// CHECK: string text = "hello world!";
def MultiLevel : Middle {
  let append items = [4];
  let append text = "!";
}

// Test nested top-level let append.
let append items = [100] in {
  let append items = [200] in {
    // CHECK: def NestedTopLevel
    // CHECK: list<int> items = [1, 2, 100, 200];
    def NestedTopLevel : Base;
  }
}

// CHECK: def OverrideAfterAppend
// CHECK: list<int> items = [10];
def OverrideAfterAppend : Base {
  let append items = [3];
  let items = [10];
}

// CHECK: def PrependDag
// CHECK: dag d = (op 0:$z);
def PrependDag : Base {
  let prepend d = (op 0:$z);
}

// CHECK: def PrependList
// CHECK: list<int> items = [0, 1, 2];
def PrependList : Base {
  let prepend items = [0];
}

// CHECK: def PrependString
// CHECK: string text = "say hello";
def PrependString : Base {
  let prepend text = "say ";
}

// Test prepend on unset fields.
// CHECK: def PrependUnset
// CHECK: list<int> vals = [1];
// CHECK: string msg = "hi";
def PrependUnset : WithUnset {
  let prepend vals = [1];
  let prepend msg = "hi";
}

// Test top-level let with append.
let append items = [100] in {
  // CHECK: def TopLevelAppend
  // CHECK: list<int> items = [1, 2, 100];
  def TopLevelAppend : Base;
}

// Test top-level let with prepend.
let prepend items = [0] in {
  // CHECK: def TopLevelPrepend
  // CHECK: list<int> items = [0, 1, 2];
  def TopLevelPrepend : Base;
}
