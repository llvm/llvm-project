
// RUN: llvm-tblgen %s | FileCheck %s

def a {
  bits<2> opc = { 0, 1 };
  bits<2> opc2 = { 1, 0 };
  bits<1> opc3 = { 1 };
  bits<2> b = { opc{0}, opc2{0} };
  bits<2> c = { opc{1}, opc2{1} };
  bits<2> c = { opc3{0}, opc3 };
}

// CHECK: def a {
// CHECK:   bits<2> opc = { 0, 1 };
// CHECK:   bits<2> opc2 = { 1, 0 };
// CHECK:   bits<1> opc3 = { 1 };
// CHECK:   bits<2> b = { 1, 0 };
// CHECK:   bits<2> c = { 1, 1 };
// CHECK: }

def {
  bits<3> B2 = 0b011;  // ok

  bits<3> C2 = 0b111;  // ok

  bits<2> D1 = { 0, 0 }; // ok
  bits<2> D2 = { 0b00 }; // ok
  bits<1> D5 = { 0 };    // ok
  bits<1> D6 = { 1 };    // ok

  bits<8> E;
  let E{7...0} = {0,0,1,?,?,?,?,?};
  let E{3...0} = 0b0010;

  bits<8> F1 = { 0, 1, 0b1001, 0, 0b0 }; // ok

  bits<8> G1 = { 0, { 1, 0b1001, 0 }, 0b0 }; // ok
  bits<8> G2 = { 0, { 1, 0b1001 }, 0, 0b0 }; // ok
  bits<8> G3 = { 0, 1, { 0b1001 }, 0, 0b0 }; // ok

  bits<16> H;
  let H{15...0} = { { 0b11001100 }, 0b00110011 };
  bits<16> I = { G1, G2 };

  // Make sure we can initialise ints with bits<> values.
  int J = H;
  int K = { 0, 1 };

  bits<2> L;
  let L{0} = 1;
  let L{1} = !eq(L{0}, 0);
}

// CHECK: def {{.*}} {
// CHECK: bits<3> B2 = { 0, 1, 1 };
// CHECK: bits<3> C2 = { 1, 1, 1 };
// CHECK: bits<2> D1 = { 0, 0 };
// CHECK: bits<2> D2 = { 0, 0 };
// CHECK: bits<1> D5 = { 0 };
// CHECK: bits<1> D6 = { 1 };
// CHECK: bits<8> E = { 0, 0, 1, ?, 0, 0, 1, 0 };
// CHECK: bits<8> F1 = { 0, 1, 1, 0, 0, 1, 0, 0 };
// CHECK: bits<8> G1 = { 0, 1, 1, 0, 0, 1, 0, 0 };
// CHECK: bits<8> G2 = { 0, 1, 1, 0, 0, 1, 0, 0 };
// CHECK: bits<8> G3 = { 0, 1, 1, 0, 0, 1, 0, 0 };
// CHECK: bits<16> H = { 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1 };
// CHECK: bits<16> I = { 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0 };
// CHECK: int J = 52275;
// CHECK: int K = 1;
// CHECK: bits<2> L = { 0, 1 };
// CHECK: }
