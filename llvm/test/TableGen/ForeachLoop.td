// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Register<string name, int idx> {
  string Name = name;
  int Index = idx;
}

// CHECK-NOT: !strconcat

foreach i = 0-3 in
  def Q#i : Register<"Q"#i, i>;

// CHECK: def Q0
// CHECK: def Q1
// CHECK: def Q2
// CHECK: def Q3

foreach i = [0, 1, 2, 3, 4, 5, 6, 7] in
  def R#i : Register<"R"#i, i>;

// CHECK: def R0
// CHECK: string Name = "R0";
// CHECK: int Index = 0;

// CHECK: def R1
// CHECK: string Name = "R1";
// CHECK: int Index = 1;

// CHECK: def R2
// CHECK: string Name = "R2";
// CHECK: int Index = 2;

// CHECK: def R3
// CHECK: string Name = "R3";
// CHECK: int Index = 3;

// CHECK: def R4
// CHECK: string Name = "R4";
// CHECK: int Index = 4;

// CHECK: def R5
// CHECK: string Name = "R5";
// CHECK: int Index = 5;

// CHECK: def R6
// CHECK: string Name = "R6";
// CHECK: int Index = 6;

// CHECK: def R7
// CHECK: string Name = "R7";
// CHECK: int Index = 7;

foreach i = {0-3,9-7} in
  def S#i : Register<"Q"#i, i>;

// CHECK: def S0
// CHECK: def S1
// CHECK: def S2
// CHECK: def S3
// CHECK: def S7
// CHECK: def S8
// CHECK: def S9
