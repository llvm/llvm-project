// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

defvar claim = "This is the end of the world!";

// CHECK: def Rec1
// CHECK:   fullNoLength = "This is the end of the world!";
// CHECK:   fullLength = "This is the end of the world!";
// CHECK:   thisIsTheEnd = "This is the end";
// CHECK:   DoorsSong = "the end";
// CHECK:   finalNoLength = "end of the world!";
// CHECK:   finalLength = "end of the world!";

def Rec1 {
  string fullNoLength = !substr(claim, 0);
  string fullLength = !substr(claim, 0, 999);
  string thisIsTheEnd = !substr(claim, 0, 15);
  string DoorsSong = !substr(claim, 8, 7);
  string finalNoLength = !substr(claim, 12);
  string finalLength = !substr(claim, 12, !sub(!size(claim), 12));
}

// CHECK: def Rec2 {
// CHECK:   lastName = "Flintstone";

def Rec2 {
  string firstName = "Fred";
  string name = firstName # " " # "Flintstone";
  string lastName = !substr(name, !add(!size(firstName), 1));
}

// CHECK: def Rec3 {
// CHECK:   test1 = "";
// CHECK:   test2 = "";
// CHECK:   test3 = "";
// CHECK:   test4 = "h";
// CHECK:   test5 = "hello";
// CHECK:   test6 = "";

def Rec3 {
  string test1 = !substr("", 0, 0);
  string test2 = !substr("", 0, 9);
  string test3 = !substr("hello", 0, 0);
  string test4 = !substr("hello", 0, 1);
  string test5 = !substr("hello", 0, 99);
  string test6 = !substr("hello", 5, 99);
}

// CHECK: def Rec4
// CHECK:   message = "This is the end of the world!";
// CHECK:   messagePrefix = "This is th...";
// CHECK:   warning = "Bad message: 'This is th...'";

class C<string msg> {
  string message = msg;
  string messagePrefix = !substr(message, 0, 10) # "...";
}

def Rec4 : C<claim> {
  string warning = "Bad message: '" # messagePrefix # "'";
}

#ifdef ERROR1

// ERROR1: expected string, got type 'int'
// ERROR1: expected int, got type 'bits<3>'
// ERROR1: expected int, got type 'string'
// ERROR1: !substr start position is out of range 0...29: 30
// ERROR1: !substr length must be nonnegative

def Rec8 {
  string claim1 = !substr(42, 0, 3);
  string claim2 = !substr(claim, 0b101);
  string claim3 = !substr(claim, 0, "oops");
}

def Rec9 {
  string claim1 = !substr(claim, !add(!size(claim), 1));
  string claim2 = !substr(claim, 0, -13);
}
#endif
