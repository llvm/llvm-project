// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s | FileCheck %s
// Checks that tablegen calculation of register unit's lanemask.
// Detail is described just after tests.

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

// Register Example (VE):
// SX0 -- SW0 (sub_i32)
//     \- SF0 (sub_f32)

def sub_i32     : SubRegIndex<32, 32>;        // Low 32 bit (32..63)
def sub_f32     : SubRegIndex<32>;            // High 32 bit (0..31)

def SW0 : MyReg<"sw0", []>;
def SF0 : MyReg<"sf0", [], [!cast<MyReg>("SW0")]>;

let SubRegIndices = [sub_i32, sub_f32] in {
def SX0 : MyReg<"s0", [SW0, SF0]>;
}

def I64 : MyClass<1, [i64], (sequence "SX%u", 0, 0)>;

def TestTarget : Target;

// CHECK: extern const LaneBitmask TestTargetLaneMaskLists[] = {
// CHECK-NEXT: /* 0 */ LaneBitmask(0x0000000000000003), LaneBitmask::getAll(),
// CHECK-NEXT: /* 2 */ LaneBitmask(0xFFFFFFFFFFFFFFFF), LaneBitmask::getAll(),

// A sub_f32 has 0x0001 as a lane bitmask.  A sub_i32 has 0x0002.
// Both are disjointed, but VE defines those are aliased registers since
// many instructions contaminate other part of register values.  So, VE
// doesn't want to allocate SF0 and SW0 simultaneously.  TableGen's
// computeRegUnitLaneMasks is optimizable using intersection algorithm,
// but it calculates 0x0000 as a intersection of 0x0001 and 0x0002.
// This causes wrong live-in calculation later.  Union mechanism can
// calculate correct lanemaks as 0x0003.
