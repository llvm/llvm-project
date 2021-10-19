// Check support for `!cond' operator as part of a `let' statement.
// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak


class C<bits<3> x, bits<4> y, bit z> {
  bits<16> n;

  let n{11}  = !cond(y{3}: 1,
                     y{2}: x{0},
                     y{1}: x{1},
                     y{0}: x{2},
                     {1} :?);
  let n{10...9}= !cond(x{2}: y{3...2},
                       x{1}: y{2...1},
                       x{1}: y{1...0},
                       {1} : ?);
  let n{8...6} = !cond(x{2}: 0b010,  1 : 0b110);
  let n{5...4} = !cond(x{1}: y{3...2}, 1 :  {0, 1});
  let n{3...0} = !cond(x{0}: y{3...0}, 1 : {z, y{2}, y{1}, y{0}});
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
