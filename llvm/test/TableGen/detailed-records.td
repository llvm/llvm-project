// RUN: llvm-tblgen -print-detailed-records %s | FileCheck %s

// This test file ensures that the DetailedRecords backend prints all
// the global variables, classes, and records.

// CHECK: ----- Global Variables
// CHECK: Answer
// CHECK: Greeting

// CHECK: ----- Classes
// CHECK: Class1
// CHECK: Class2
// CHECK: Class3
// CHECK-NEXT: Template args:
// CHECK-NEXT:   int Class3:Arg = ?
// CHECK-NEXT: Superclasses: (none)
// CHECK-NEXT: Fields: (none)

// CHECK: ----- Records
// CHECK: ""
// CHECK: Info
// CHECK: Rec1
// CHECK: anonymous_0

defvar Greeting = "Hello there.";
defvar Answer = 42;

class Class1 {
}

class Class2 {
}

class Class3<int Arg> {
}

def : Class1;

def "" : Class2;

def Rec1;

def Info : Class1;

