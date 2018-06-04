// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Honorific<string t> {
  string honorific = t;
}

def Mr : Honorific<"Mr.">;
def Ms : Honorific<"Ms.">;
def Mrs : Honorific<"Mrs.">;
def TVAR : Honorific<"Bogus">;

class Name<string n, Honorific t> {
  string name = n;
  Honorific honorific = t;
}

class AName<string name, Honorific honorific> : 
  Name<!subst("FIRST", "John", !subst("LAST", "Smith", name)),
       !subst(TVAR, Mr, honorific)>;

def JohnSmith : AName<"FIRST LAST", TVAR>;
def JaneSmith : AName<"Jane LAST", Ms>;
def JohnSmithJones : AName<"FIRST LAST-Jones", Mr>;
def JimmyJohnson : AName<"Jimmy Johnson", Mr>;

// CHECK:      ------------- Classes -----------------
// CHECK-NEXT: class AName<string AName:name = ?, Honorific AName:honorific = ?> {
// CHECK-NEXT:   string name = !subst("FIRST", "John", !subst("LAST", "Smith", AName:name));
// CHECK-NEXT:   Honorific honorific = !subst(TVAR, Mr, AName:honorific);
// CHECK-NEXT: }
// CHECK-NEXT: class Honorific<string Honorific:t = ?> {
// CHECK-NEXT:   string honorific = Honorific:t;
// CHECK-NEXT: }
// CHECK-NEXT: class Name<string Name:n = ?, Honorific Name:t = ?> {
// CHECK-NEXT:   string name = Name:n;
// CHECK-NEXT:   Honorific honorific = Name:t;
// CHECK-NEXT: }
// CHECK-NEXT: ------------- Defs -----------------
// CHECK-NEXT: def JaneSmith {
// CHECK-NEXT:   string name = "Jane Smith";
// CHECK-NEXT:   Honorific honorific = Ms;
// CHECK-NEXT: }
// CHECK-NEXT: def JimmyJohnson {
// CHECK-NEXT:   string name = "Jimmy Johnson";
// CHECK-NEXT:   Honorific honorific = Mr;
// CHECK-NEXT: }
// CHECK-NEXT: def JohnSmith {
// CHECK-NEXT:   string name = "John Smith";
// CHECK-NEXT:   Honorific honorific = Mr;
// CHECK-NEXT: }
// CHECK-NEXT: def JohnSmithJones {
// CHECK-NEXT:   string name = "John Smith-Jones";
// CHECK-NEXT:   Honorific honorific = Mr;
// CHECK-NEXT: }
// CHECK-NEXT: def Mr 
// CHECK-NEXT:   string honorific = "Mr.";
// CHECK-NEXT: }
// CHECK-NEXT: def Mrs {
// CHECK-NEXT:   string honorific = "Mrs.";
// CHECK-NEXT: }
// CHECK-NEXT: def Ms {
// CHECK-NEXT:   string honorific = "Ms.";
// CHECK-NEXT: }
// CHECK-NEXT: def TVAR {
// CHECK-NEXT:   string honorific = "Bogus";
// CHECK-NEXT: }
