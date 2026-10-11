// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
include "llvm/Target/Target.td"

class MyReg<string n> : Register<n> { let Namespace = "Test"; }
class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> { let Size = size; }

def sub_lo : SubRegIndex<32, 0>;
// Erroneous: the high half is declared 64 bits at offset 32, so the pair's
// sub-registers cover [0, 96) bits, past the end of the 64-bit register.
def sub_hi : SubRegIndex<64, 32>;

def L0 : MyReg<"l0">;
def H0 : MyReg<"h0">;
def L1 : MyReg<"l1">;
def H1 : MyReg<"h1">;

def LoRegs : MyClass<32, [i32], (add L0, L1)>;
def HiRegs : MyClass<32, [i32], (add H0, H1)>;

let SubRegIndices = [sub_lo, sub_hi], CoveredBySubRegs = 1 in {
  def P0 : MyReg<"p0"> { let SubRegs = [L0, H0]; }
  def P1 : MyReg<"p1"> { let SubRegs = [L1, H1]; }
}
def PairRegs : MyClass<64, [i64], (add P0, P1)>;

def TestTarget : Target;

// CHECK: error: register 'P0' has size 64 but its explicit sub-registers cover 96 bits
