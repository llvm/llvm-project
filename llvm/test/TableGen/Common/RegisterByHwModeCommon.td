include "llvm/Target/Target.td"

/// A minimal reduced version of the RISC-V RVY register variants where pointers
/// use either Xn or Xn_Y registers depending on CapMode and 64-bit predicates.
/// Define HWModes for the full cross-product here since we can only match one
/// HWMode at any given time.
def Is32Bit : Predicate<"!Subtarget->is64Bit()">;
def Is64Bit : Predicate<"Subtarget->is64Bit()">;
def UseYRegForPtr : Predicate<"Subtarget->useYRegForPtr()">;
def UseXRegForPtr : Predicate<"!Subtarget->useYRegForPtr()">;
defvar XPtr32 = DefaultMode;
def XPtr64 : HwMode<[Is64Bit, UseXRegForPtr]>;
def YPtr32 : HwMode<[Is32Bit, UseYRegForPtr]>;
def YPtr64 : HwMode<[Is64Bit, UseYRegForPtr]>;

class MyReg<string n> : Register<n> {
  let Namespace = "MyTarget";
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def X2 : MyReg<"x2">;
def X3 : MyReg<"x3">;

def Y0 : MyReg<"y0">;
def Y1 : MyReg<"y1">;
def Y2 : MyReg<"y2">;
def Y3 : MyReg<"y3">;

def XLenVT : ValueTypeByHwMode<[XPtr32, XPtr64, YPtr32, YPtr64],
                               [i32,    i64,    i32,    i64]>;
def YLenVT : ValueTypeByHwMode<[XPtr32, XPtr64, YPtr32, YPtr64],
                               [c64,    c128,   c64,    c128]>;
def PtrVT : ValueTypeByHwMode<[XPtr32, XPtr64, YPtr32, YPtr64],
                              [XLenVT, XLenVT, YLenVT, YLenVT]>;
defvar RegInfo32 = RegInfo<32,32,32>;
defvar RegInfo64 = RegInfo<64,64,64>;
def XLenRI : RegInfoByHwMode<[XPtr32,    XPtr64,    YPtr32,    YPtr64],
                             [RegInfo32, RegInfo32, RegInfo64, RegInfo64]>;
def XRegs : RegisterClass<"MyTarget", [XLenVT], 32, (add X0, X1, X2, X3)> {
  let RegInfos = XLenRI;  // Needed to determine size of registers
}
defvar RegInfo128 = RegInfo<128,128,128>;
def YLenRI : RegInfoByHwMode<[XPtr32,    XPtr64,    YPtr32,     YPtr64],
                             [RegInfo64, RegInfo64, RegInfo128, RegInfo128]>;

def YRegs : RegisterClass<"MyTarget", [YLenVT], 64, (add Y0, Y1, Y2, Y3)> {
  let RegInfos = YLenRI;  // Needed to determine size of registers
}
def PtrRC : RegClassByHwMode<[XPtr32, XPtr64, YPtr32, YPtr64],
                             [XRegs,  XRegs,  YRegs,  YRegs]>;

def PtrRegOperand : RegisterOperand<PtrRC>;

def NullReg : RegisterByHwMode<PtrRC, [XPtr32, XPtr64, YPtr32, YPtr64],
                                      [X0,     X0,     Y0,     Y0]>;

class TestInstruction : Instruction {
  let Size = 2;
  let Namespace = "MyTarget";
  let hasSideEffects = false;
  let hasExtraSrcRegAllocReq = false;
  let hasExtraDefRegAllocReq = false;

  field bits<16> Inst;
  bits<3> dst;
  bits<3> src;
  bits<3> opcode;

  let Inst{2-0} = dst;
  let Inst{5-3} = src;
  let Inst{7-5} = opcode;
}

def TEST_XREG : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "test_x $dst, $src";
  let opcode = 0;
}
def TEST_YREG : TestInstruction {
  let OutOperandList = (outs YRegs:$dst);
  let InOperandList = (ins YRegs:$src);
  let AsmString = "test_y $dst, $src";
  let opcode = 1;
}
def TEST_PTRREG : TestInstruction {
  let OutOperandList = (outs PtrRegOperand:$dst);
  let InOperandList = (ins PtrRegOperand:$src);
  let AsmString = "test_ptr $dst, $src";
  let opcode = 2;
}
