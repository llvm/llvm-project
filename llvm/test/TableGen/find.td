// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// This file contains tests for the !find bang operator.

defvar Sentence = "This is the end of the world.";

// CHECK: def Rec01
// CHECK-NEXT:   int FirstThe = 8
// CHECK-NEXT:   int SameThe = 8
// CHECK-NEXT:   int SecondThe = 19
// CHECK-NEXT:   int ThirdThe = -1

def Rec01 {
  int FirstThe = !find(Sentence, "the");
  int SameThe = !find(Sentence, "the", FirstThe);
  int SecondThe = !find(Sentence, "the", !add(FirstThe, 1));
  int ThirdThe = !find(Sentence, "the", !add(SecondThe, 1));
}

class C1<string name> {
  string Name = name;
  bit IsJr = !ne(!find(name, "Jr"), -1);
}

// CHECK: def Rec02
// CHECK-NEXT:   string Name = "Sally Smith"
// CHECK-NEXT:   bit IsJr = 0
// CHECK: def Rec03
// CHECK-NEXT:   string Name = "Fred Jones, Jr."
// CHECK-NEXT:   bit IsJr = 1

def Rec02 : C1<"Sally Smith">;
def Rec03 : C1<"Fred Jones, Jr.">;

// CHECK: def Rec04
// CHECK-NEXT:   int ThisPos = 0
// CHECK-NEXT:   int WorldPos = 23
// CHECK-NEXT:   int TooLong = -1

def Rec04 {
  int ThisPos = !find(Sentence, "This");
  int WorldPos = !find(Sentence, "world.");
  int TooLong = !find(Sentence, "world.country");
}

// CHECK: def Rec05
// CHECK-NEXT:   string Name = "Pat Snork"
// CHECK-NEXT:   bit IsJr = 0
// CHECK-NEXT:   bit JrOrSnork = 1

def Rec05 : C1<"Pat Snork"> {
  bit JrOrSnork = !or(IsJr, !ne(!find(Name, "Snork"), -1));
}

#ifdef ERROR1

// ERROR1: !find start position is out of range 0...29: 100

def Rec06 {
  int Test1 = !find(Sentence, "the", 100);
}
#endif

