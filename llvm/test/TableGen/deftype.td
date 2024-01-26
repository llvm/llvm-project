// RUN: llvm-tblgen %s | FileCheck %s

deftype Byte = bits<8>;
deftype IntList = list<int>;
deftype ByteList = list<Byte>;
class Class<Byte b> {
  Byte byte = b;
}
deftype ClassList = list<Class>;

// CHECK:      def test {
// CHECK-NEXT:   bits<8> byte = { 0, 1, 1, 1, 1, 0, 1, 1 };
// CHECK-NEXT:   list<int> ints = [1, 2, 3];
// CHECK-NEXT:   list<bits<8>> bytes = [{ 0, 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 1, 1 }];
// CHECK-NEXT:   list<Class> defs = [anonymous_0, anonymous_1, anonymous_2];
// CHECK-NEXT: }
def test {
  Byte byte = 123;
  IntList ints = [1, 2, 3];
  ByteList bytes = [1, 2, 3];
  ClassList defs = [Class<1>, Class<2>, Class<3>];
}
