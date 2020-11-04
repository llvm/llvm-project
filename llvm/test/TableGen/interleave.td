// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

defvar EmptyList = []<string>;
defvar OneList = ["hello"];
defvar StringList = ["foo", "bar", "zoo", "snork", "quux"];
defvar IntList = [0, 1, 2, 3, 4, 5, 6, 7];
defvar BitsList = [ {0, 1, 0}, {1, 1, 1}, {0, 0, 1} ];
defvar BitList = [0, 1, 1, 0, 1]<bit>;

class Ishify<list<string> words> {
  list<string> ret = !foreach(w, words, w # "ify");
}

// CHECK: def Rec1
// CHECK:   Test1 = "";
// CHECK:   Test2 = "hello";
// CHECK:   Test3 = "foobarzoosnorkquux";
// CHECK:   Test4 = "foo, bar, zoo, snork, quux";
// CHECK:   Test5 = "foo & bar & zoo & snork & quux & grits";

def Rec1 {
  string Test1 = !interleave(EmptyList, "/");
  string Test2 = !interleave(OneList, ":");
  string Test3 = !interleave(StringList, "");
  string Test4 = !interleave(StringList, ", ");
  string Test5 = !interleave(!listconcat(StringList, ["grits"]), " & ");
}

// CHECK: def Rec2
// CHECK: Test1 = "01234567";
// CHECK: Test2 = "0, 1, 2, 3, 4, 5, 6, 7";
// CHECK: Test3 = "0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 42";

def Rec2 {
  string Test1 = !interleave(IntList, "");
  string Test2 = !interleave(IntList, ", ");
  string Test3 = !interleave(!listconcat(IntList, [42]), " & ");
}

// CHECK: def Rec3
// CHECK: Test1 = "271";
// CHECK: Test2 = "2, 7, 1";
// CHECK: Test3 = "2 & 7 & 1 & 0";

def Rec3 {
  string Test1 = !interleave(BitsList, "");
  string Test2 = !interleave(BitsList, ", ");
  string Test3 = !interleave(!listconcat(BitsList, [ {0, 0, 0} ]), " & ");
}

// CHECK: def Rec4
// CHECK: Test1 = "01101";
// CHECK: Test2 = "0, 1, 1, 0, 1";
// CHECK: Test3 = "0 and 1 and 1 and 0 and 1 and 1";

def Rec4 {
  string Test1 = !interleave(BitList, "");
  string Test2 = !interleave(BitList, ", ");
  string Test3 = !interleave(!listconcat(BitList, [1]), " and ");
}

// CHECK: def Rec5
// CHECK: Colors = ["red", "green", "yellow"];
// CHECK: ColorList = "redify, greenify, yellowify";

def Rec5 {
  list<string> Colors = ["red", "green", "yellow"];
  string ColorList = !interleave(Ishify<Colors>.ret, ", ");
}

#ifdef ERROR1
def op;

// ERROR1: expected list of string, int, bits, or bit; got value of type

def Rec6 {
  string Bad = !interleave([(op), (op "hello")], " = ");
}
#endif
