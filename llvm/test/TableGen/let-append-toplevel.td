// RUN: llvm-tblgen %s | FileCheck %s

// Test 'let append' and 'let prepend' syntax in top-level let.

class Base {
  list<int> items = [1, 2];
  string text = "hello";
}

// Test that 'append' and 'prepend' can be used as field names in top-level let.
class HasAppendField {
  list<int> append = [1, 2];
  list<int> prepend = [3, 4];
  int other = 0;
}

// Test top-level let with multiple items: no mode, append, and prepend.
// CHECK: def MultiItemTopLevel
// CHECK: list<int> items = [0, 1, 2, 100];
// CHECK: string text = "prefix hello suffix";
let prepend items = [0], append items = [100],
    prepend text = "prefix ", append text = " suffix" in {
  def MultiItemTopLevel : Base;
}

// Test nested top-level let append.
let append items = [100] in {
  let append items = [200] in {
    // CHECK: def NestedTopLevel
    // CHECK: list<int> items = [1, 2, 100, 200];
    def NestedTopLevel : Base;
  }
}

// Test top-level let with append.
let append items = [100] in {
  // CHECK: def TopLevelAppend
  // CHECK: list<int> items = [1, 2, 100];
  def TopLevelAppend : Base;
}

// Test top-level 'let append = ...' where 'append' is the field name.
let append = [10, 20] in {
  // CHECK: def TopLevelFieldNamedAppend
  // CHECK: list<int> append = [10, 20];
  // CHECK: list<int> prepend = [3, 4];
  def TopLevelFieldNamedAppend : HasAppendField;
}

// Test top-level 'let prepend = ...' where 'prepend' is the field name.
let prepend = [30, 40] in {
  // CHECK: def TopLevelFieldNamedPrepend
  // CHECK: list<int> append = [1, 2];
  // CHECK: list<int> prepend = [30, 40];
  def TopLevelFieldNamedPrepend : HasAppendField;
}

// Test top-level let with prepend.
let prepend items = [0] in {
  // CHECK: def TopLevelPrepend
  // CHECK: list<int> items = [0, 1, 2];
  def TopLevelPrepend : Base;
}
