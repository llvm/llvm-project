// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Register<string name, int idx> {
  string Name = name;
  int Index = idx;
}

foreach i = [0, 1, 2, 3, 4, 5, 6, 7] in {
  def R#i : Register<"R"#i, i>;
  def F#i : Register<"F"#i, i>;
}

// CHECK: def F0
// CHECK: string Name = "F0";
// CHECK: int Index = 0;

// CHECK: def F1
// CHECK: string Name = "F1";
// CHECK: int Index = 1;

// CHECK: def F2
// CHECK: string Name = "F2";
// CHECK: int Index = 2;

// CHECK: def F3
// CHECK: string Name = "F3";
// CHECK: int Index = 3;

// CHECK: def F4
// CHECK: string Name = "F4";
// CHECK: int Index = 4;

// CHECK: def F5
// CHECK: string Name = "F5";
// CHECK: int Index = 5;

// CHECK: def F6
// CHECK: string Name = "F6";
// CHECK: int Index = 6;

// CHECK: def F7
// CHECK: string Name = "F7";
// CHECK: int Index = 7;

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
