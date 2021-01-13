// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// This file tests the error message that is printed when a body is
// terminated with a semicolon in addition to the close brace.

// CHECK: class Class0
// CHECK: def Rec0

class Class0 {
}

def Rec0 {
}

multiclass MC0 {
  def R;
}

#ifdef ERROR1

// ERROR1: error: A class or def body should not end with a semicolon
// ERROR1: Semicolon ignored
// ERROR1: error: A class or def body should not end with a semicolon
// ERROR1: Semicolon ignored
// ERROR1: error: A multiclass body should not end with a semicolon
// ERROR1: Semicolon ignored

class Class1 {
};

def Rec1 {
};

multiclass MC1 {
  def R;
};

#endif
