// Mixed bits (with a field reference) and integer arms in !cond.
// Selecting the bits arm used to crash when converting to the unified int type.
// RUN: llvm-tblgen %s | FileCheck %s

class Profile<bit x> {
  bit xx = x;
}

def P1 : Profile<1>;
def P0 : Profile<0>;

class TT<Profile p> {
  bits<4> reg = {0, 1, 0, 1};
  bits<2> fmt;
  let fmt = !cond(!eq(p.xx, 1) : {0, reg{0}},
                  true : 3);
}

// Unset register bits, matching the original crash report.
class UU<Profile p> {
  bits<10> reg;
  bits<2> fmt = !cond(!eq(p.xx, 1) : {0, reg{0}},
                      true : 3);
}

def TakenBits : TT<P1>;
def TakenInt : TT<P0>;
def UnsetBits : UU<P1>;
def UnsetInt : UU<P0>;

// CHECK: def TakenBits
// CHECK-NEXT: bits<4> reg = { 0, 1, 0, 1 };
// CHECK-NEXT: bits<2> fmt = { 0, 1 };

// CHECK: def TakenInt
// CHECK-NEXT: bits<4> reg = { 0, 1, 0, 1 };
// CHECK-NEXT: bits<2> fmt = { 1, 1 };

// CHECK: def UnsetBits
// CHECK-NEXT: bits<10> reg = { ?, ?, ?, ?, ?, ?, ?, ?, ?, ? };
// CHECK-NEXT: bits<2> fmt = { 0, reg{0} };

// CHECK: def UnsetInt
// CHECK-NEXT: bits<10> reg = { ?, ?, ?, ?, ?, ?, ?, ?, ?, ? };
// CHECK-NEXT: bits<2> fmt = { 1, 1 };
