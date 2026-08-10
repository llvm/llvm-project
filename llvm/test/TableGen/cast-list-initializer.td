// RUN: llvm-tblgen %s | FileCheck %s

class Foo<bits<8> b> {
// CHECK: list<int> ListOfInts = [170];
// CHECK: list<int> AnotherList = [170, 7];
  list<int> ListOfInts = [!cast<int>(b)];
  list<int> AnotherList = [!cast<int>(b), !cast<int>({1, 1, 1})];
}

def : Foo<{1, 0, 1, 0, 1, 0, 1, 0}>;
