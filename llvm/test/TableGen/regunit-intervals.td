// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
  let RegistersAreIntervals  = 1;
}

def sub_lo : SubRegIndex<32>;
def sub_hi : SubRegIndex<32, 32>;

let Namespace = "Test" in {
  // Simple 32-bit registers (each gets 1 regunit)
  def R0 : Register<"r0">;
  def R1 : Register<"r1">;
  def R2 : Register<"r2">;
  def R3 : Register<"r3">;

  // 64-bit register composed of R0:R1 (gets 2 regunits)
  def R0_R1 : Register<"r0_r1"> {
    let SubRegs = [R0, R1];
    let SubRegIndices = [sub_lo, sub_hi];
  }

  // 64-bit register composed of R2:R3 (gets 2 regunits)
  def R2_R3 : Register<"r2_r3"> {
    let SubRegs = [R2, R3];
    let SubRegIndices = [sub_lo, sub_hi];
  }
}

// CHECK: extern const unsigned TestTargetRegUnitIntervals[][2] = {
// CHECK-NEXT: { 0, 1 },
// CHECK-NEXT: { 1, 2 },
// CHECK-NEXT: { 2, 3 },
// CHECK-NEXT: { 3, 4 },

let Namespace = "Test" in {
  def R4 : Register<"r4">;      // Gets unit 4
  def R5 : Register<"r5">;      // Gets unit 5
  def R6 : Register<"r6">;      // Gets unit 6
  def R7 : Register<"r7">;      // Gets unit 7
  
  // This register skips R5, creating non-contiguous units {4, 6}
  def R4_R6 : Register<"r4_r6"> {
    let SubRegs = [R4, R6];
    let SubRegIndices = [sub_lo, sub_hi];
  }

  // This register skips R6, creating non-contiguous units {5, 7}
  def R5_R7 : Register<"r4_r6"> {
    let SubRegs = [R5, R7];
    let SubRegIndices = [sub_lo, sub_hi];
  }
}


def GPR32 : RegisterClass<"Test", [i32], 32, (add R0, R1, R2, R3)>;
def GPR64 : RegisterClass<"Test", [i64], 64, (add R0_R1, R2_R3)>;

// Note R5 is assigned 6,7 so that R6 gets 5,6
// CHECK-NEXT: { 4, 5 },
// CHECK-NEXT: { 6, 7 },
// CHECK-NEXT: { 5, 6 },
// CHECK-NEXT: { 7, 8 },
// All contiguous
// CHECK-NEXT: { 0, 2 },
// CHECK-NEXT: { 2, 4 },
// CHECK-NEXT: { 4, 6 },
// CHECK-NEXT: { 6, 8 },
// CHECK-NEXT: };
