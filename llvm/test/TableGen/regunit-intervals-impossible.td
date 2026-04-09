// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
  let RegistersAreIntervals = 1;
}

def sub_lo : SubRegIndex<32>;
def sub_hi : SubRegIndex<32, 32>;

let Namespace = "Test" in {
  def R1 : Register<"r1">;  // unit 0
  def R2 : Register<"r2">;  // unit 1
  def R3 : Register<"r3">;  // unit 2

  // First composite: units {0, 1} - contiguous, OK
  def R1_R2 : Register<"r1_r2"> {
    let SubRegs = [R1, R2];
    let SubRegIndices = [sub_lo, sub_hi];
  }

  // Second composite: units {0, 2} - non-contiguous!
  // Algorithm will swap 1 and 2, making R1_R2 not contiguous
  def R1_R3 : Register<"r1_r3"> {
    let SubRegs = [R1, R3];
    let SubRegIndices = [sub_lo, sub_hi];
  }
}

def GPR32 : RegisterClass<"Test", [i32], 32, (add R1, R2, R3)>;

// CHECK: error: cannot enforce regunit intervals, final renumbering did not produce contiguous units for register R1_R2
