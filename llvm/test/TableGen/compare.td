// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s

// This file tests the comparison bang operators.

class BitCompare<bit a, bit b> {
  list<bit> compare = [!eq(a, b), !ne(a, b),
                       !lt(a, b), !le(a, b),
                       !gt(a, b), !ge(a, b)];
}

class BitsCompare<bits<3> a, bits<3> b> {
  list<bit> compare = [!eq(a, b), !ne(a, b),
                       !lt(a, b), !le(a, b),
                       !gt(a, b), !ge(a, b)];
}

class IntCompare<int a, int b> {
  list<bit> compare = [!eq(a, b), !ne(a, b),
                       !lt(a, b), !le(a, b),
                       !gt(a, b), !ge(a, b)];
}

class StringCompare<string a, string b> {
  list<bit> compare = [!eq(a, b), !ne(a, b),
                       !lt(a, b), !le(a, b),
                       !gt(a, b), !ge(a, b)];
}

multiclass MC {
  def _MC;
}

// CHECK: def Bit00
// CHECK:   compare = [1, 0, 0, 1, 0, 1];
// CHECK: def Bit01
// CHECK:   compare = [0, 1, 1, 1, 0, 0];
// CHECK: def Bit10
// CHECK:   compare = [0, 1, 0, 0, 1, 1];
// CHECK: def Bit11
// CHECK:   compare = [1, 0, 0, 1, 0, 1];

def Bit00 : BitCompare<0, 0>;
def Bit01 : BitCompare<0, 1>;
def Bit10 : BitCompare<1, 0>;
def Bit11 : BitCompare<1, 1>;

// CHECK: def Bits1
// CHECK:   compare = [0, 1, 1, 1, 0, 0];
// CHECK: def Bits2
// CHECK:   compare = [1, 0, 0, 1, 0, 1];
// CHECK: def Bits3
// CHECK:   compare = [0, 1, 0, 0, 1, 1];

def Bits1 : BitsCompare<{0, 1, 0}, {1, 0, 1}>;
def Bits2 : BitsCompare<{0, 1, 1}, {0, 1, 1}>;
def Bits3 : BitsCompare<{1, 1, 1}, {0, 1, 1}>;

// CHECK: def Int1
// CHECK:   compare = [0, 1, 1, 1, 0, 0];
// CHECK: def Int2
// CHECK:   compare = [1, 0, 0, 1, 0, 1];
// CHECK: def Int3
// CHECK:   compare = [0, 1, 0, 0, 1, 1];

def Int1 : IntCompare<-7, 13>;
def Int2 : IntCompare<42, 42>;
def Int3 : IntCompare<108, 42>;

// CHECK: def Record1
// CHECK:   compare1 = [1, 0];
// CHECK:   compare2 = [0, 1];
// CHECK:   compare3 = [1, 1];

defm foo : MC;
defm bar : MC;

def Record1 {
  list<bit> compare1 = [!eq(Bit00, Bit00), !eq(Bit00, Bit01)];
  list<bit> compare2 = [!ne(Bit00, Bit00), !ne(Bit00, Int1)];
  list<bit> compare3 = [!eq(bar_MC, bar_MC), !ne(bar_MC, foo_MC)];
}

// CHECK: def String1
// CHECK:   compare = [0, 1, 1, 1, 0, 0];
// CHECK: def String2
// CHECK:   compare = [1, 0, 0, 1, 0, 1];
// CHECK: def String3
// CHECK:   compare = [0, 1, 0, 0, 1, 1];
// CHECK: def String4
// CHECK:   compare = [0, 1, 0, 0, 1, 1];
def String1 : StringCompare<"bar", "foo">;
def String2 : StringCompare<"foo", "foo">;
def String3 : StringCompare<"foo", "bar">;
def String4 : StringCompare<"foo", "Foo">;

#ifdef ERROR1

// ERROR1: expected bit, bits, int, string, or record; got value

def Zerror1 {
  bit compare1 = !eq([0, 1, 2], [0, 1, 2]);
}

#endif

#ifdef ERROR2

// ERROR2: expected bit, bits, int, or string; got value

def Zerror2 {
  bit compare1 = !lt(Bit00, Bit00);
}

#endif

