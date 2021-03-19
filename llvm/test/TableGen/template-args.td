// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not llvm-tblgen -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not llvm-tblgen -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s

// This file tests that template arguments are type-checked and cast
// if necessary.

// Class template arguments.

class Class1<string nm> {
  string Name = nm;
}

// CHECK: def Rec1
// CHECK:   string Name = "Alice"
// CHECK:   string NameName = "AliceAlice"

def Rec1 : Class1<"Alice"> {
  string NameName = Name # Name;
}

#ifdef ERROR1
// ERROR1: Value specified for template argument 'Class1:nm' (#0) is of type int

def Rec2 : Class1<42> {
}
#endif

class Class2<bits<8> cd> {
  int Code = cd;
}

// CHECK: def Rec3
// CHECK:   int Code = 42
// CHECK:   list<int> CodeList = [42]

def Rec3 : Class2<0b00101010> {
  list<int> CodeList = [Code];
}

// CHECK: def Rec4
// CHECK:   int Code = 42
// CHECK:   list<int> CodeList = [42]

def Rec4 : Class2<42> {
  list<int> CodeList = [Code];
}

#ifdef ERROR2
// ERROR2: Value specified for template argument 'Class2:cd' (#0) is of type string

def Rec5 : Class2<"oops"> {
  list<int> CodeList = [Code];
}
#endif

// Anonymous class instantiation template arguments.

// CHECK: def Rec6
// CHECK:   string Name = "Ted"

def Rec6 {
  string Name = Class1<"Ted">.Name;
}

#ifdef ERROR3
// ERROR3: Value specified for template argument 'Class1:nm' (#0) is of type int

def Rec7 {
  string Name = Class1<42>.Name;
}
#endif

// CHECK: def Rec8
// CHECK:   list<int> CodeList = [42]

def Rec8 {
  list<int> CodeList = [Class2<42>.Code];
}

#ifdef ERROR4
// ERROR4: Value specified for template argument 'Class2:cd' (#0) is of type string

def Rec9 {
  list<int> CodeList = [Class2<"huh?">.Code];
}
#endif

// Multiclass template arguments.

multiclass MC1<string nm> {
  def _1 {
    string Name = nm;
  }
  def _2 {
    string NameNmae = nm # nm;
  }
}

// CHECK: def RecMC1_1
// CHECK:   string Name = "Carol"
// CHECK: def RecMC1_2
// CHECK:   string NameNmae = "CarolCarol"

defm RecMC1 : MC1<"Carol">;

#ifdef ERROR5
// ERROR5: Value specified for template argument 'MC1::nm' (#0) is of type int

defm RecMC2 : MC1<42>;
#endif

multiclass MC2<bits<8> cd> {
  def _1 {
    bits<8> Code = cd;
  }
  def _2 {
    int Code = cd;
  }
  def _3 {
    list<int> CodeList = [cd];
  }
}

// CHECK: def RecMC3_1
// CHECK:   bits<8> Code = { 0, 0, 1, 0, 1, 0, 1, 0 }
// CHECK: def RecMC3_2
// CHECK:   int Code = 42
// CHECK: def RecMC3_3
// CHECK:   list<int> CodeList = [42]

defm RecMC3 : MC2<42>;

#ifdef ERROR6
// ERROR6: Value specified for template argument 'MC2::cd' (#0) is of type string

defm RecMC4 : MC2<"Bob">;
#endif
