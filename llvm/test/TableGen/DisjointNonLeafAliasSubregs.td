// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK
// Checks that tablegen calculation of register unit's lanemask.
// It covers the scenario, where non-leaf aliasing registers are used to create 
// disjoint subregs for defining new register (this case inspired from VE tablegen)
// This test is inspired via hypothetical extrapolating VE's registers backend.
//
// This situation is identical to the test DisjoinLeafAliasSubregs.td, except the 
// fact, that here the alias registers are not only used as mutually disjoint subregs
// to define new register, but also these alias registers are not necessarily the
// the leaf registers, implying they themselves are made out of subregs.
//
// Here also, the lanemask corresponding to unique regUnit will be accurately tell
// regUnit position, whereas the laneMask for common regunit, accounting for aliasing
// , could be still arbitrary based on the exact super register its part of.

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

def sub_i16     : SubRegIndex<16, 16>;        // High 16 bit (16..31)
def sub_f16     : SubRegIndex<16>;            // Low 16 bit (0..15)
def sub_i32     : SubRegIndex<32, 32>;        // High 32 bit (32..63)
def sub_f32     : SubRegIndex<32>;            // Low 32 bit (0..31)

// Registers SW0 & SF0 are aliases

// VE's Registers Extrapolated example:
// SX0 -- SW0 (sub_f32) -- SW0_LO16 (sub_f16)
//    \                 \- SW0_HI16 (sub_i16)  
//     \- SF0 (sub_i32)
def SW0_LO16 : MyReg<"sw0_lo16", []>;
def SW0_HI16 : MyReg<"sw0_hi16", []>;

let SubRegIndices = [sub_f16, sub_i16] in {
  def SW0 : MyReg<"sw1", [SW0_LO16, SW0_HI16]>;
}

def SF0 : MyReg<"sf0", [], [!cast<MyReg>("SW0")]>;

let SubRegIndices = [sub_f32, sub_i32] in {
  def SX0 : MyReg<"s0", [!cast<MyReg>("SW0"), !cast<MyReg>("SF0")]>;
}

def I64 : MyClass<1, [i64], (add (sequence "SX%u", 0, 0))>;

def TestTarget : Target;

// CHECK:      RegisterClass I64:
// CHECK:      	LaneMask: 0000000000000007
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	CoveredBySubRegs: 1
// CHECK:      	Regs: SX0
// CHECK:      	SubClasses: I64
// CHECK:      	SuperClasses:

// CHECK:      SubRegIndex sub_f16:
// CHECK-NEXT: 	LaneMask: 0000000000000001
// CHECK:      SubRegIndex sub_f32:
// CHECK-NEXT: 	LaneMask: 0000000000000003
// CHECK:      SubRegIndex sub_i16:
// CHECK-NEXT: 	LaneMask: 0000000000000002
// CHECK:      SubRegIndex sub_i32:
// CHECK-NEXT: 	LaneMask: 0000000000000004

// CHECK:      Register SF0:
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	Native RegUnit:  SF0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF

// CHECK:      Register SW0:
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	SubReg sub_f16 = SW0_LO16
// CHECK:      	RegUnit SW0_LO16 :  SW0_LO16
// CHECK:      	SubReg sub_i16 = SW0_HI16
// CHECK:      	RegUnit SW0_HI16 :  SW0_HI16
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF
// CHECK:      	Native RegUnit:  SW0_LO16
// CHECK:      	LaneMask Value: 0000000000000001
// CHECK:      	Native RegUnit:  SW0_HI16
// CHECK:      	LaneMask Value: 0000000000000002

// CHECK:      Register SX0:
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 1
// CHECK:      	SubReg sub_f16 = SW0_LO16
// CHECK:      	RegUnit SW0_LO16 :  SW0_LO16
// CHECK:      	SubReg sub_f32 = SW0
// CHECK:      	RegUnit SW0 :  SF0~SW0
// CHECK:      	RegUnit SW0 :  SW0_LO16
// CHECK:      	RegUnit SW0 :  SW0_HI16
// CHECK:      	SubReg sub_i16 = SW0_HI16
// CHECK:      	RegUnit SW0_HI16 :  SW0_HI16
// CHECK:      	SubReg sub_i32 = SF0
// CHECK:      	RegUnit SF0 :  SF0
// CHECK:      	RegUnit SF0 :  SF0~SW0
// CHECK:      	Native RegUnit:  SF0
// CHECK:      	LaneMask Value: 0000000000000004
// CHECK:      	Native RegUnit:  SF0~SW0
// CHECK:      	LaneMask Value: 0000000000000004
// CHECK:      	Native RegUnit:  SW0_LO16
// CHECK:      	LaneMask Value: 0000000000000001
// CHECK:      	Native RegUnit:  SW0_HI16
// CHECK:      	LaneMask Value: 0000000000000002

// CHECK:      Register SW0_HI16:
// CHECK:      	CostPerUse: 0 
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	Native RegUnit:  SW0_HI16
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF

// CHECK:      Register SW0_LO16:
// CHECK:      	CostPerUse: 0 
// CHECK:      	CoveredBySubregs: 1
// CHECK:      	HasDisjunctSubRegs: 0
// CHECK:      	Native RegUnit:  SW0_LO16
// CHECK:      	LaneMask Value: FFFFFFFFFFFFFFFF