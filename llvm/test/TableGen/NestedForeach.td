// RUN: llvm-tblgen %s | FileCheck %s

class Droid<string series, int release, string model, int patchlevel> {
  string Series = series;
  int Release = release;
  string Model = model;
  int Patchlevel = patchlevel;
}

foreach S = ["R", "C"] in {
  foreach R = [2, 3, 4] in {
    foreach M = ["D", "P", "Q"] in {
      foreach P = [0, 2, 4] in {
        def S#R#M#P : Droid<S, R, M, P>;
      }
    }
  }
}

// CHECK: def C2D0
// CHECK: def C2D2
// CHECK: def C2D4
// CHECK: def C2P0
// CHECK: def C2P2
// CHECK: def C2P4
// CHECK: def C2Q0
// CHECK: def C2Q2
// CHECK: def C2Q4
// CHECK: def C3D0
// CHECK: def C3D2
// CHECK: def C3D4
// CHECK: def C3P0
// CHECK: def C3P2
// CHECK: def C3P4
// CHECK: def C3Q0
// CHECK: def C3Q2
// CHECK: def C3Q4
// CHECK: def C4D0
// CHECK: def C4D2
// CHECK: def C4D4
// CHECK: def C4P0
// CHECK: def C4P2
// CHECK: def C4P4
// CHECK: def C4Q0
// CHECK: def C4Q2
// CHECK: def C4Q4
// CHECK: def R2D0
// CHECK: def R2D2
// CHECK: def R2D4
// CHECK: def R2P0
// CHECK: def R2P2
// CHECK: def R2P4
// CHECK: def R2Q0
// CHECK: def R2Q2
// CHECK: def R2Q4
// CHECK: def R3D0
// CHECK: def R3D2
// CHECK: def R3D4
// CHECK: def R3P0
// CHECK: def R3P2
// CHECK: def R3P4
// CHECK: def R3Q0
// CHECK: def R3Q2
// CHECK: def R3Q4
// CHECK: def R4D0
// CHECK: def R4D2
// CHECK: def R4D4
// CHECK: def R4P0
// CHECK: def R4P2
// CHECK: def R4P4
// CHECK: def R4Q0
// CHECK: def R4Q2
// CHECK: def R4Q4
