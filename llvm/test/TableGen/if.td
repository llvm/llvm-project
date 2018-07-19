// RUN: llvm-tblgen %s | FileCheck %s
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
  let n{10-9}= !if(x{2}, y{3-2},
               !if(x{1}, y{2-1},
               !if(x{0}, y{1-0}, ?)));
  let n{8-6} = !if(x{2}, 0b010, 0b110);
  let n{5-4} = !if(x{1}, y{3-2}, {0, 1});
  let n{3-0} = !if(x{0}, y{3-0}, {z, y{2}, y{1}, y{0}});
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
