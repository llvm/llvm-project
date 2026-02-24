// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

// The HwModeSelect class is intended to serve as a base class for other
// classes that are then used to select a value based on the HW mode.
// It contains a list of HW modes, and a derived class should provide a
// list of corresponding values.
// These two lists must have the same size. Make sure that a violation of
// this requirement is diagnosed.

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def TestReg : Register<"testreg">;
def TestClass : RegisterClass<"TestTarget", [i32], 32, (add TestReg)>;

def HasFeat1 : Predicate<"Subtarget->hasFeat1()">;
def HasFeat2 : Predicate<"Subtarget->hasFeat2()">;

def TestMode1 : HwMode<[HasFeat1]>;
def TestMode2 : HwMode<[HasFeat2]>;

// CHECK: error: assertion failed: The Objects and Modes lists must be the same length
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def BadDef : ValueTypeByHwMode<[TestMode1, TestMode2, DefaultMode],
                               [i8, i16, i32, i64]>;

