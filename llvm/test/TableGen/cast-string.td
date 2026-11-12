// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// This file tests the !cast bang operator with the result type string.

defvar IntList = [0, 1, 2, 3, 4, 5, 6];

// CHECK: def Rec0
// CHECK:   string str3 = "a string here"

def Rec0 {
  string str = "a string";
  string str2 = !cast<string>(str);
  string str3 = !cast<string>(str # " here");
}

// CHECK: def Rec1
// CHECK:   string str = "42 -108"

def Rec1 {
  int int1 = 42;
  int int2 = -108;
  string str = !cast<string>(int1) # " " # !cast<string>(int2);
}

// CHECK: def Rec2
// CHECK:   string str = "0, 1"

def Rec2 {
  bit bit1 = false;
  bit bit2 = true;
  string str = !cast<string>(bit1) # ", " # !cast<string>(bit2);
}

// CHECK: def Rec3
// CHECK:   string str = "5 and 37"

def Rec3 {
  bits<4> bits1 = 0b0101;
  bits<8> bits2 = 0b00100101;
  string str = !cast<string>(bits1) # " and " # !cast<string>(bits2);
}

// CHECK: def Rec4
// CHECK:   string str = "Rec1, Rec2"

def Rec4 {
  string str = !cast<string>(Rec1) # ", " # !cast<string>(Rec2);
}

#ifdef ERROR1

// ERROR1: nitializer of 'str' in 'Rec5' could not be fully resolved

def Rec5 {
  string str = !cast<string>(IntList);
}

#endif
