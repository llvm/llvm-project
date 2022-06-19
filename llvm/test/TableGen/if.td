// RUN: llvm-tblgen --no-warn-on-unused-template-args %s | FileCheck %s
// XFAIL: vg_leak

// Support for an `!if' operator as part of a `let' statement.
// CHECK:      class C
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, !if({ C:y{3} }, 1, !if({ C:y{2} }, { C:x{0} }, !if({ C:y{1} }, { C:x{1} }, !if({ C:y{0} }, { C:x{2} }, ?)))){0}, !if({ C:x{2} }, { C:y{3}, C:y{2} }, !if({ C:x{1} }, { C:y{2}, C:y{1} }, !if({ C:x{0} }, { C:y{1}, C:y{0} }, ?))){1}, !if({ C:x{2} }, { C:y{3}, C:y{2} }, !if({ C:x{1} }, { C:y{2}, C:y{1} }, !if({ C:x{0} }, { C:y{1}, C:y{0} }, ?))){0}, !if({ C:x{2} }, { 0, 1, 0 }, { 1, 1, 0 }){2}, !if({ C:x{2} }, { 0, 1, 0 }, { 1, 1, 0 }){1}, !if({ C:x{2} }, { 0, 1, 0 }, { 1, 1, 0 }){0}, !if({ C:x{1} }, { C:y{3}, C:y{2} }, { 0, 1 }){1}, !if({ C:x{1} }, { C:y{3}, C:y{2} }, { 0, 1 }){0}, !if({ C:x{0} }, { C:y{3}, C:y{2}, C:y{1}, C:y{0} }, { C:z, C:y{2}, C:y{1}, C:y{0} }){3}, !if({ C:x{0} }, { C:y{3}, C:y{2}, C:y{1}, C:y{0} }, { C:z, C:y{2}, C:y{1}, C:y{0} }){2}, !if({ C:x{0} }, { C:y{3}, C:y{2}, C:y{1}, C:y{0} }, { C:z, C:y{2}, C:y{1}, C:y{0} }){1}, !if({ C:x{0} }, { C:y{3}, C:y{2}, C:y{1}, C:y{0} }, { C:z, C:y{2}, C:y{1}, C:y{0} }){0} };
class C<bits<3> x, bits<4> y, bit z> {
  bits<16> n;

  let n{11}  = !if(y{3}, 1,
               !if(y{2}, x{0},
               !if(y{1}, x{1},
               !if(y{0}, x{2}, ?))));
  let n{10...9}= !if(x{2}, y{3...2},
                 !if(x{1}, y{2...1},
                 !if(x{0}, y{1...0}, ?)));
  let n{8...6} = !if(x{2}, 0b010, 0b110);
  let n{5...4} = !if(x{1}, y{3...2}, {0, 1});
  let n{3...0} = !if(x{0}, y{3...0}, {z, y{2}, y{1}, y{0}});
}

def C1 : C<{1, 0, 1}, {0, 1, 0, 1}, 0>;
def C2 : C<{0, 1, 0}, {1, 0, 1, 0}, 1>;
def C3 : C<{0, 0, 0}, {1, 0, 1, 0}, 0>;
def C4 : C<{0, 0, 0}, {0, 0, 0, 0}, 0>;

// CHECK: def C1
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1 };
// CHECK: def C2
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0 };
// CHECK: def C3
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, 1, ?, ?, 1, 1, 0, 0, 1, 0, 0, 1, 0 };
// CHECK: def C4
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, ?, ?, ?, 1, 1, 0, 0, 1, 0, 0, 0, 0 };

class S<int s> {
  bits<2> val = !if(!eq(s, 8),  {0, 0},
                !if(!eq(s, 16), 0b01,
                !if(!eq(s, 32), 2,
                !if(!eq(s, 64), {1, 1}, ?))));
}

def D8  : S<8>;
def D16 : S<16>;
def D32 : S<32>;
def D64 : S<64>;
def D128: S<128>;
// CHECK: def D128
// CHECK-NEXT: bits<2> val = { ?, ? };
// CHECK: def D16
// CHECK-NEXT: bits<2> val = { 0, 1 };
// CHECK: def D32
// CHECK-NEXT: bits<2> val = { 1, 0 };
// CHECK: def D64
// CHECK-NEXT: bits<2> val = { 1, 1 };
// CHECK: def D8
// CHECK-NEXT: bits<2> val = { 0, 0 };

// Make sure !if gets propagated across multiple layers of inheritance.
class getInt<int c> {
  int ret = !if(c, 0, 1);
}
class I1<int c> {
  int i = getInt<c>.ret;
}
class I2<int c> : I1<c>;

// CHECK: def DI1 {     // I1
// CHECK-NEXT: int i = 0;
def DI1: I1<1>;

// CHECK: def DI2 {     // I1 I2
// CHECK-NEXT: int i = 0;
def DI2: I2<1>;

// Check that !if with operands of different subtypes can initialize a
// supertype variable.
//
// CHECK: def EXd1 {
// CHECK:   E x = E1d;
// CHECK: }
//
// CHECK: def EXd2 {
// CHECK:   E x = E2d;
// CHECK: }
class E<int dummy> {}
class E1<int dummy> : E<dummy> {}
class E2<int dummy> : E<dummy> {}

class EX<int cc, E1 b, E2 c> {
  E x = !if(cc, b, c);
}

def E1d : E1<0>;
def E2d : E2<0>;

def EXd1 : EX<1, E1d, E2d>;
def EXd2 : EX<0, E1d, E2d>;

// CHECK: def Not1
// CHECK: Result1a = "OK"
// CHECK: Result1b = "OK"
// CHECK: Result1c = "OK"
// CHECK: Result1d = "OK"
// CHECK: Result1e = "OK"
// CHECK: def Not2
// CHECK: Result2a = "OK"
// CHECK: Result2b = "OK"
// CHECK: Result2c = "OK"
// CHECK: Result2d = "OK"

def Not1 {
  bit True = 1;
  string Result1a = !if(True, "OK", "not OK");
  string Result1b = !if(!not(True), "not OK", "OK");

  bit False = 0;
  string Result1c = !if(False, "not OK", "OK");
  string Result1d = !if(!not(False), "OK", "not OK");
  string Result1e = !if(!not(!not(False)), "not OK", "OK");
}

def Not2 {
  int one = 1;
  string Result2a = !if(one, "OK", "not OK");
  string Result2b = !if(!not(one), "not OK", "OK");

  int zero = 0;
  string Result2c = !if(zero, "not OK", "OK");
  string Result2d = !if(!not(zero), "OK", "not OK");
}

// CHECK:      def One
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [1, 2, 3];

// CHECK:      def OneB
// CHECK-NEXT: list<int> vals = [1, 2, 3];

// CHECK:      def Two
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [4, 5, 6];

// CHECK:      def TwoB
// CHECK-NEXT: list<int> vals = [4, 5, 6];

class A<list<list<int>> vals> {
  list<int> first = vals[0];
  list<int> rest  = !if(!empty(!tail(vals)), vals[0], vals[1]);
}

def One : A<[[1,2,3]]>;
def Two : A<[[1,2,3], [4,5,6]]>;

class B<list<int> v> {
  list<int> vals = v;
}

class BB<list<list<int>> vals> : B<!if(!empty(!tail(vals)), vals[0], vals[1])>;
class BBB<list<list<int>> vals> : BB<vals>;

def OneB : BBB<[[1,2,3]]>;
def TwoB : BBB<[[1,2,3],[4,5,6]]>;
