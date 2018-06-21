// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Instr<int i> {
  int index = i;
}

multiclass Test {
  def Vx#NAME#PS : Instr<0>;
  def Vx#NAME#PD : Instr<1>;
  def Vy#NAME#PS : Instr<2>;
  def Vy#NAME#PD : Instr<3>;
}

defm ADD : Test;
defm SUB : Test;

// CHECK: VxADDPD
// CHECK: index = 1;
// CHECK: VxADDPS
// CHECK: index = 0;

// CHECK: VxSUBPD
// CHECK: index = 1;
// CHECK: VxSUBPS
// CHECK: index = 0;

// CHECK: VyADDPD
// CHECK: index = 3;
// CHECK: VyADDPS
// CHECK: index = 2;

// CHECK: VySUBPD
// CHECK: index = 3;
// CHECK: VySUBPS
// CHECK: index = 2;
