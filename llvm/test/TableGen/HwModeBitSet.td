// This is to test the scenario where different HwMode attributes coexist.
// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-REG
// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SUBTARGET


include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def TestMode : HwMode<"+feat", []>;
def TestMode1 : HwMode<"+feat1", []>;
def TestMode2 : HwMode<"+feat2", []>;

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

def ValueModeVT : ValueTypeByHwMode<[DefaultMode, TestMode, TestMode1],
                                    [i32,  i64, f32]>;

let RegInfos = RegInfoByHwMode<[DefaultMode, TestMode],
                               [RegInfo<32,32,32>, RegInfo<64,64,64>]> in
def XRegs : MyClass<32, [ValueModeVT], (sequence "X%u", 0, 15)>;

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

// Modes who are not controlling Register related features will be manipulated
// the same as DefaultMode.
// CHECK-REG-LABEL: RegisterClass XRegs:
// CHECK-REG: SpillSize: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }
// CHECK-REG: SpillAlignment: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }
// CHECK-REG: Regs: X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15

// CHECK-REG-LABEL: RegisterClass XPairsClass:
// CHECK-REG: SpillSize: { Default:64 TestMode:128 TestMode1:64 TestMode2:64 }
// CHECK-REG: SpillAlignment: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }
// CHECK-REG: CoveredBySubRegs: 1
// CHECK-REG: Regs: X0_X1 X2_X3 X4_X5 X6_X7 X8_X9 X10_X11 X12_X13 X14_X15

// CHECK-REG-LABEL: SubRegIndex sub_even:
// CHECK-REG: Offset: { Default:0 TestMode:0 TestMode1:0 TestMode2:0 }
// CHECK-REG: Size: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }
// CHECK-REG-LABEL: SubRegIndex sub_odd:
// CHECK-REG: Offset: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }
// CHECK-REG: Size: { Default:32 TestMode:64 TestMode1:32 TestMode2:32 }

//============================================================================//
//--------------------- Encoding/Decoding parts ------------------------------//
//============================================================================//
def fooTypeEncDefault : InstructionEncoding {
  let Size = 8;
  field bits<64> SoftFail = 0;
  bits<64> Inst;
  bits<8> factor;
  let Inst{7...0} = factor;
  let Inst{3...2} = 0b10;
  let Inst{1...0} = 0b00;
}

def fooTypeEncA : InstructionEncoding {
  let Size = 4;
  field bits<32> SoftFail = 0;
  bits<32> Inst;
  bits<8> factor;
  let Inst{7...0} = factor;
  let Inst{3...2} = 0b11;
  let Inst{1...0} = 0b00;
}


def foo : Instruction {
  bits<32> Inst;
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let EncodingInfos = EncodingByHwMode<
    [TestMode2, DefaultMode], [fooTypeEncA, fooTypeEncDefault]
  >;
  let AsmString = "foo  $factor";
}

// CHECK-SUBTARGET-LABEL: unsigned TestTargetGenSubtargetInfo::getHwModeSet() const {
// CHECK-SUBTARGET:         unsigned Modes = 0;
// CHECK-SUBTARGET:         if (checkFeatures("+feat")) Modes |= (1 << 0);
// CHECK-SUBTARGET:         if (checkFeatures("+feat1")) Modes |= (1 << 1);
// CHECK-SUBTARGET:         if (checkFeatures("+feat2")) Modes |= (1 << 2);
// CHECK-SUBTARGET:         return Modes;
// CHECK-SUBTARGET:       }
// CHECK-SUBTARGET-LABEL: unsigned TestTargetGenSubtargetInfo::getHwMode(enum HwModeType type) const {
// CHECK-SUBTARGET:         unsigned Modes = getHwModeSet();
// CHECK-SUBTARGET:         if (!Modes)
// CHECK-SUBTARGET:           return Modes;
// CHECK-SUBTARGET:         switch (type) {
// CHECK-SUBTARGET:         case HwMode_Default:
// CHECK-SUBTARGET:           return llvm::countr_zero(Modes) + 1;
// CHECK-SUBTARGET:         case HwMode_ValueType:
// CHECK-SUBTARGET:           Modes &= 3;
// CHECK-SUBTARGET:           if (!Modes)
// CHECK-SUBTARGET:             return Modes;
// CHECK-SUBTARGET:           if (!llvm::has_single_bit<unsigned>(Modes))
// CHECK-SUBTARGET:             llvm_unreachable("Two or more HwModes for ValueType were found!");
// CHECK-SUBTARGET:           return llvm::countr_zero(Modes) + 1;
// CHECK-SUBTARGET:         case HwMode_RegInfo:
// CHECK-SUBTARGET:           Modes &= 1;
// CHECK-SUBTARGET:           if (!Modes)
// CHECK-SUBTARGET:             return Modes;
// CHECK-SUBTARGET:           if (!llvm::has_single_bit<unsigned>(Modes))
// CHECK-SUBTARGET:             llvm_unreachable("Two or more HwModes for RegInfo were found!");
// CHECK-SUBTARGET:           return llvm::countr_zero(Modes) + 1;
// CHECK-SUBTARGET:         case HwMode_EncodingInfo:
// CHECK-SUBTARGET:           Modes &= 4;
// CHECK-SUBTARGET:           if (!Modes)
// CHECK-SUBTARGET:             return Modes;
// CHECK-SUBTARGET:           if (!llvm::has_single_bit<unsigned>(Modes))
// CHECK-SUBTARGET:             llvm_unreachable("Two or more HwModes for EncodingInfo were found!");
// CHECK-SUBTARGET:           return llvm::countr_zero(Modes) + 1;
// CHECK-SUBTARGET:         }
// CHECK-SUBTARGET:         llvm_unreachable("unexpected HwModeType");
// CHECK-SUBTARGET:         return 0; // should not get here
// CHECK-SUBTARGET:       }

