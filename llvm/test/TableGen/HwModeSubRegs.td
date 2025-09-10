// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s
include "llvm/Target/Target.td"

def HasFeat : Predicate<"Subtarget->hasFeat()">;

def TestMode : HwMode<[HasFeat]>;

class MyReg<string n>
  : Register<n> {
  let Namespace = "Test";
}
class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> {
  let Size = size;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def X2 : MyReg<"x2">;
def X3 : MyReg<"x3">;
def X4 : MyReg<"x4">;
def X5 : MyReg<"x5">;
def X6 : MyReg<"x6">;
def X7 : MyReg<"x7">;
def X8 : MyReg<"x8">;
def X9 : MyReg<"x9">;
def X10 : MyReg<"x10">;
def X11 : MyReg<"x11">;
def X12 : MyReg<"x12">;
def X13 : MyReg<"x13">;
def X14 : MyReg<"x14">;
def X15 : MyReg<"x15">;

def ModeVT : ValueTypeByHwMode<[DefaultMode, TestMode],
                               [i32,  i64]>;
let RegInfos = RegInfoByHwMode<[DefaultMode, TestMode],
                               [RegInfo<32,32,32>, RegInfo<64,64,64>]> in
def XRegs : MyClass<32, [ModeVT], (sequence "X%u", 0, 15)>;

def sub_even : SubRegIndex<32> {
  let SubRegRanges = SubRegRangeByHwMode<[DefaultMode, TestMode],
                                         [SubRegRange<32>, SubRegRange<64>]>;
}
def sub_odd  : SubRegIndex<32, 32> {
  let SubRegRanges = SubRegRangeByHwMode<[DefaultMode, TestMode],
                                         [SubRegRange<32, 32>, SubRegRange<64, 64>]>;
}

def XPairs : RegisterTuples<[sub_even, sub_odd],
                            [(decimate (rotl XRegs, 0), 2),
                             (decimate (rotl XRegs, 1), 2)]>;

let RegInfos = RegInfoByHwMode<[DefaultMode, TestMode],
                               [RegInfo<64,64,32>, RegInfo<128,128,64>]> in
def XPairsClass : MyClass<64, [untyped], (add XPairs)>;

def TestTarget : Target;

// CHECK-LABEL: RegisterClass XRegs:
// CHECK: SpillSize: { Default:32 TestMode:64 }
// CHECK: SpillAlignment: { Default:32 TestMode:64 }
// CHECK: Regs: X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15

// CHECK-LABEL: RegisterClass XPairsClass:
// CHECK: SpillSize: { Default:64 TestMode:128 }
// CHECK: SpillAlignment: { Default:32 TestMode:64 }
// CHECK: CoveredBySubRegs: 1
// CHECK: Regs: X0_X1 X2_X3 X4_X5 X6_X7 X8_X9 X10_X11 X12_X13 X14_X15

// CHECK-LABEL: SubRegIndex sub_even:
// CHECK: Offset: { Default:0 TestMode:0 }
// CHECK: Size: { Default:32 TestMode:64 }
// CHECK-LABEL: SubRegIndex sub_odd:
// CHECK: Offset: { Default:32 TestMode:64 }
// CHECK: Size: { Default:32 TestMode:64 }
