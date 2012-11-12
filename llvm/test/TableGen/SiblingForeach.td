// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Set<int i = 0, int j = 0, int k = 0> {
  int I = i;
  int J = j;
  int K = k;
}

foreach i = [1, 2, 3] in {
  def I1_#i : Set<i>;
  foreach j = [1, 2, 3] in {
    def I1_#i#_J1_#j : Set<i, j>;
  }
  def I2_#i : Set<i>;
  foreach j = [4, 5, 6] in {
    foreach k = [1, 2, 3] in {
      def I3_#i#_J2_#j#_K1_#k : Set<i, j, k>;
    }
    def I4_#i#_J3_#j : Set<i, j>;
  }
}

// CHECK: def I1_1
// CHECK: int I = 1;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I1_1_J1_1
// CHECK: int I = 1;
// CHECK: int J = 1;
// CHECK: int K = 0;

// CHECK: def I1_1_J1_2
// CHECK: int I = 1;
// CHECK: int J = 2;
// CHECK: int K = 0;

// CHECK: def I1_1_J1_3
// CHECK: int I = 1;
// CHECK: int J = 3;
// CHECK: int K = 0;

// CHECK: def I1_2
// CHECK: int I = 2;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I1_2_J1_1
// CHECK: int I = 2;
// CHECK: int J = 1;
// CHECK: int K = 0;

// CHECK: def I1_2_J1_2
// CHECK: int I = 2;
// CHECK: int J = 2;
// CHECK: int K = 0;

// CHECK: def I1_2_J1_3
// CHECK: int I = 2;
// CHECK: int J = 3;
// CHECK: int K = 0;

// CHECK: def I1_3
// CHECK: int I = 3;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I1_3_J1_1
// CHECK: int I = 3;
// CHECK: int J = 1;
// CHECK: int K = 0;

// CHECK: def I1_3_J1_2
// CHECK: int I = 3;
// CHECK: int J = 2;
// CHECK: int K = 0;

// CHECK: def I1_3_J1_3
// CHECK: int I = 3;
// CHECK: int J = 3;
// CHECK: int K = 0;

// CHECK: def I2_1
// CHECK: int I = 1;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I2_2
// CHECK: int I = 2;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I2_3
// CHECK: int I = 3;
// CHECK: int J = 0;
// CHECK: int K = 0;

// CHECK: def I3_1_J2_4_K1_1
// CHECK: int I = 1;
// CHECK: int J = 4;
// CHECK: int K = 1;

// CHECK: def I3_1_J2_4_K1_2
// CHECK: int I = 1;
// CHECK: int J = 4;
// CHECK: int K = 2;

// CHECK: def I3_1_J2_4_K1_3
// CHECK: int I = 1;
// CHECK: int J = 4;
// CHECK: int K = 3;

// CHECK: def I3_1_J2_5_K1_1
// CHECK: int I = 1;
// CHECK: int J = 5;
// CHECK: int K = 1;

// CHECK: def I3_1_J2_5_K1_2
// CHECK: int I = 1;
// CHECK: int J = 5;
// CHECK: int K = 2;

// CHECK: def I3_1_J2_5_K1_3
// CHECK: int I = 1;
// CHECK: int J = 5;
// CHECK: int K = 3;

// CHECK: def I3_1_J2_6_K1_1
// CHECK: int I = 1;
// CHECK: int J = 6;
// CHECK: int K = 1;

// CHECK: def I3_1_J2_6_K1_2
// CHECK: int I = 1;
// CHECK: int J = 6;
// CHECK: int K = 2;

// CHECK: def I3_1_J2_6_K1_3
// CHECK: int I = 1;
// CHECK: int J = 6;
// CHECK: int K = 3;

// CHECK: def I3_2_J2_4_K1_1
// CHECK: int I = 2;
// CHECK: int J = 4;
// CHECK: int K = 1;

// CHECK: def I3_2_J2_4_K1_2
// CHECK: int I = 2;
// CHECK: int J = 4;
// CHECK: int K = 2;

// CHECK: def I3_2_J2_4_K1_3
// CHECK: int I = 2;
// CHECK: int J = 4;
// CHECK: int K = 3;

// CHECK: def I3_2_J2_5_K1_1
// CHECK: int I = 2;
// CHECK: int J = 5;
// CHECK: int K = 1;

// CHECK: def I3_2_J2_5_K1_2
// CHECK: int I = 2;
// CHECK: int J = 5;
// CHECK: int K = 2;

// CHECK: def I3_2_J2_5_K1_3
// CHECK: int I = 2;
// CHECK: int J = 5;
// CHECK: int K = 3;

// CHECK: def I3_2_J2_6_K1_1
// CHECK: int I = 2;
// CHECK: int J = 6;
// CHECK: int K = 1;

// CHECK: def I3_2_J2_6_K1_2
// CHECK: int I = 2;
// CHECK: int J = 6;
// CHECK: int K = 2;

// CHECK: def I3_2_J2_6_K1_3
// CHECK: int I = 2;
// CHECK: int J = 6;
// CHECK: int K = 3;

// CHECK: def I3_3_J2_4_K1_1
// CHECK: int I = 3;
// CHECK: int J = 4;
// CHECK: int K = 1;

// CHECK: def I3_3_J2_4_K1_2
// CHECK: int I = 3;
// CHECK: int J = 4;
// CHECK: int K = 2;

// CHECK: def I3_3_J2_4_K1_3
// CHECK: int I = 3;
// CHECK: int J = 4;
// CHECK: int K = 3;

// CHECK: def I3_3_J2_5_K1_1
// CHECK: int I = 3;
// CHECK: int J = 5;
// CHECK: int K = 1;

// CHECK: def I3_3_J2_5_K1_2
// CHECK: int I = 3;
// CHECK: int J = 5;
// CHECK: int K = 2;

// CHECK: def I3_3_J2_5_K1_3
// CHECK: int I = 3;
// CHECK: int J = 5;
// CHECK: int K = 3;

// CHECK: def I3_3_J2_6_K1_1
// CHECK: int I = 3;
// CHECK: int J = 6;
// CHECK: int K = 1;

// CHECK: def I3_3_J2_6_K1_2
// CHECK: int I = 3;
// CHECK: int J = 6;
// CHECK: int K = 2;

// CHECK: def I3_3_J2_6_K1_3
// CHECK: int I = 3;
// CHECK: int J = 6;
// CHECK: int K = 3;

// CHECK: def I4_1_J3_4
// CHECK: int I = 1;
// CHECK: int J = 4;
// CHECK: int K = 0;

// CHECK: def I4_1_J3_5
// CHECK: int I = 1;
// CHECK: int J = 5;
// CHECK: int K = 0;

// CHECK: def I4_1_J3_6
// CHECK: int I = 1;
// CHECK: int J = 6;
// CHECK: int K = 0;

// CHECK: def I4_2_J3_4
// CHECK: int I = 2;
// CHECK: int J = 4;
// CHECK: int K = 0;

// CHECK: def I4_2_J3_5
// CHECK: int I = 2;
// CHECK: int J = 5;
// CHECK: int K = 0;

// CHECK: def I4_2_J3_6
// CHECK: int I = 2;
// CHECK: int J = 6;
// CHECK: int K = 0;

// CHECK: def I4_3_J3_4
// CHECK: int I = 3;
// CHECK: int J = 4;
// CHECK: int K = 0;

// CHECK: def I4_3_J3_5
// CHECK: int I = 3;
// CHECK: int J = 5;
// CHECK: int K = 0;

// CHECK: def I4_3_J3_6
// CHECK: int I = 3;
// CHECK: int J = 6;
// CHECK: int K = 0;
