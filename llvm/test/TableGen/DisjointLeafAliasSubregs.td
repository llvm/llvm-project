// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK
// Checks that tablegen calculation of register unit's lanemask.
// It covers the scenario, where leaf aliasing registers are used to create 
// disjoint subregs for defining new register (this case inspired from VE tablegen)
//
// For each such leaf register, a unique regUnit will be created that should rightfully
// denote its position in the super register via laneMask. whereas, both of will also 
// have a common regUnit that accurately represents the ad hoc aliasing, even though its
// laneMask could be arbitrary which is still better situation.   

include "llvm/Target/Target.td"

class MyReg<string n, list<Register> subregs = [], list<Register> aliases = []>
  : Register<n> {
  let Namespace = "Test";
  let SubRegs = subregs;
  let Aliases = aliases;
  let CoveredBySubRegs = 1;
}
class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> {
  let Size = size;
}

def sub_i32     : SubRegIndex<32, 32>;        // High 32 bit (32..63)
def sub_f32     : SubRegIndex<32>;            // Low 32 bit (0..31)

// Registers SW0 & SF0 are aliases

// VE's Registers Example:
// SX0 -- SW0 (sub_f32)
//     \- SF0 (sub_i32)
def SW0 : MyReg<"sw0", []>;

def SF0 : MyReg<"sf0", [], [!cast<MyReg>("SW0")]>;

let SubRegIndices = [sub_f32, sub_i32] in {
  def SX0 : MyReg<"s0", [!cast<MyReg>("SW0"), !cast<MyReg>("SF0")]>;
}

def I64 : MyClass<1, [i64], (add (sequence "SX%u", 0, 0))>;

def TestTarget : Target;

// CHECK:      RegisterClass I64:
// CHECK:      	LaneMask: 0000000000000003
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	CoveredBySubRegs: 1
// CHECK:      	Regs: SX0
// CHECK:      	SubClasses: I64
// CHECK:      	SuperClasses:

// CHECK:      SubRegIndex sub_f32:
// CHECK-NEXT: 	LaneMask: 0000000000000001
// CHECK:      SubRegIndex sub_i32:
// CHECK-NEXT: 	LaneMask: 0000000000000002

// CHECK:      Register SF0:
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	Native RegUnit:  SF0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF

// CHECK:      Register SW0:
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF
// CHECK:      	Native RegUnit:  SW0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF

// CHECK:      Register SX0:
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	SubReg sub_f32 = SW0
// CHECK:      	RegUnit SW0 :  SF0~SW0
// CHECK:      	RegUnit SW0 :  SW0
// CHECK:      	SubReg sub_i32 = SF0
// CHECK:      	RegUnit SF0 :  SF0
// CHECK:      	RegUnit SF0 :  SF0~SW0
// CHECK:      	Native RegUnit:  SF0
// CHECK:      	LaneMask Value: 0000000000000002
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: 0000000000000000
// CHECK:      	Native RegUnit:  SW0
// CHECK:      	LaneMask Value: 0000000000000001