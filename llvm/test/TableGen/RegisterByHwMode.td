// RUN: llvm-tblgen --gen-subtarget -I %p/../../include -I %S %s -o - | FileCheck --check-prefix=SUBTARGET %s
// RUN: llvm-tblgen --gen-instr-info -I %p/../../include -I %S %s -o - | FileCheck --check-prefix=INSTRINFO %s
// RUN: llvm-tblgen --gen-asm-matcher -I %p/../../include -I %S %s -o - | FileCheck --check-prefix=ASMMATCHER %s
// RUN: llvm-tblgen --gen-pseudo-lowering -I %p/../../include -I %S %s -o - | FileCheck --check-prefix=PSEUDO %s
// RUN: llvm-tblgen --gen-asm-writer -I %p/../../include -I %S %s -o - | FileCheck --check-prefix=ASMWRITER %s
/// The RegInfo output is split over multiple files:
// RUN: llvm-tblgen --gen-register-info -I %p/../../include -I %S %s -o %t.inc
// RUN: FileCheck %s --input-file=%tEnums.inc --check-prefix=REGINFO-HEADER
// RUN: FileCheck %s --input-file=%tMCDesc.inc --check-prefix=REGINFO-MCDESC
/// Note: No impact on disassembler (handled by the alias expansion), so not tested here
/// Note: DAGIsel is not supported yet
// RUNTODO: llvm-tblgen --gen-dag-isel -I %p/../../include -I %S %s -o -
// RUNTODO: llvm-tblgen --gen-global-isel -I %p/../../include -I %S %s -o -

// REGINFO-HEADER-LABEL: // Registers by HwMode
// REGINFO-HEADER-NEXT: class MCRegister;
// REGINFO-HEADER-NEXT: namespace MyTarget::RegisterByHwMode {
// REGINFO-HEADER-EMPTY:
// REGINFO-HEADER-NEXT: LLVM_READONLY MCRegister getModeCountReg(unsigned HwMode);
// REGINFO-HEADER-NEXT: LLVM_READONLY MCRegister getNullReg(unsigned HwMode);
// REGINFO-HEADER-NEXT: LLVM_READONLY MCRegister getPtrRegFor64BitModesOnly(unsigned HwMode);
// REGINFO-HEADER-EMPTY:
// REGINFO-HEADER-NEXT: } // namespace MyTarget::RegisterByHwMode
// REGINFO-HEADER-EMPTY:
// REGINFO-HEADER-NEXT: } // namespace llvm

// REGINFO-MCDESC-LABEL: // Registers by HwMode
// REGINFO-MCDESC-NEXT: namespace MyTarget::RegisterByHwMode {
// REGINFO-MCDESC-EMPTY:
// REGINFO-MCDESC-NEXT: LLVM_READONLY MCRegister getModeCountReg(unsigned HwMode) {
// REGINFO-MCDESC-NEXT:   switch (HwMode) {
// REGINFO-MCDESC-NEXT:   case 0: return MyTarget::X0; // DefaultMode
// REGINFO-MCDESC-NEXT:   case 1: return MyTarget::X1; // XPtr64
// REGINFO-MCDESC-NEXT:   case 2: return MyTarget::X2; // YPtr32
// REGINFO-MCDESC-NEXT:   case 3: return MyTarget::X3; // YPtr64
// REGINFO-MCDESC-NEXT:   default: llvm_unreachable("Unhandled HwMode for Register ModeCountReg");
// REGINFO-MCDESC-NEXT:   }
// REGINFO-MCDESC-NEXT: }
// REGINFO-MCDESC-NEXT: LLVM_READONLY MCRegister getNullReg(unsigned HwMode) {
// REGINFO-MCDESC-NEXT:   switch (HwMode) {
// REGINFO-MCDESC-NEXT:   case 0: return MyTarget::X0; // DefaultMode
// REGINFO-MCDESC-NEXT:   case 1: return MyTarget::X0; // XPtr64
// REGINFO-MCDESC-NEXT:   case 2: return MyTarget::Y0; // YPtr32
// REGINFO-MCDESC-NEXT:   case 3: return MyTarget::Y0; // YPtr64
// REGINFO-MCDESC-NEXT:   default: llvm_unreachable("Unhandled HwMode for Register NullReg");
// REGINFO-MCDESC-NEXT:   }
// REGINFO-MCDESC-NEXT: }
// REGINFO-MCDESC-NEXT: LLVM_READONLY MCRegister getPtrRegFor64BitModesOnly(unsigned HwMode) {
// REGINFO-MCDESC-NEXT:   switch (HwMode) {
// REGINFO-MCDESC-NEXT:   case 1: return MyTarget::X1; // XPtr64
// REGINFO-MCDESC-NEXT:   case 3: return MyTarget::Y1; // YPtr64
// REGINFO-MCDESC-NEXT:   default: llvm_unreachable("Unhandled HwMode for Register PtrRegFor64BitModesOnly");
// REGINFO-MCDESC-NEXT:   }
// REGINFO-MCDESC-NEXT: }
// REGINFO-MCDESC-EMPTY:
// REGINFO-MCDESC-NEXT: } // namespace MyTarget::RegisterByHwMode
// REGINFO-MCDESC-EMPTY:
// REGINFO-MCDESC-NEXT: } // namespace llvm

// SUBTARGET-LABEL:  enum class MyTargetHwModeBits : unsigned {
// SUBTARGET-NEXT:    DefaultMode = 0,
// SUBTARGET-NEXT:    XPtr64 = (1 << 0),
// SUBTARGET-NEXT:    YPtr32 = (1 << 1),
// SUBTARGET-NEXT:    YPtr64 = (1 << 2),
// SUBTARGET-EMPTY:
// SUBTARGET-NEXT:    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/YPtr64),
// SUBTARGET-NEXT:  };
// SUBTARGET-NEXT:  unsigned getHwModeSet() const final;
// SUBTARGET-NEXT:  unsigned getHwMode(enum HwModeType type = HwMode_Default) const final;

// SUBTARGET-LABEL: unsigned MyTargetGenSubtargetInfo::getHwModeSet() const {
// SUBTARGET{LITERAL}:[[maybe_unused]] const auto *Subtarget =
// SUBTARGET-NEXT:        static_cast<const MyTargetSubtarget *>(this);
// SUBTARGET-NEXT:  // Collect HwModes and store them as a bit set.
// SUBTARGET-NEXT:  unsigned Modes = 0;
// SUBTARGET-NEXT:  if ((Subtarget->is64Bit()) && (!Subtarget->useYRegForPtr())) Modes |= (1 << 0);
// SUBTARGET-NEXT:  if ((!Subtarget->is64Bit()) && (Subtarget->useYRegForPtr())) Modes |= (1 << 1);
// SUBTARGET-NEXT:  if ((Subtarget->is64Bit()) && (Subtarget->useYRegForPtr())) Modes |= (1 << 2);
// SUBTARGET-NEXT:  return Modes;
// SUBTARGET-NEXT:}

// INSTRINFO-LABEL: extern const MyTargetInstrTable MyTargetDescs
// INSTRINFO:      { MyTarget::EvenPtrRC, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO-NEXT: { MyTarget::EvenXRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO-NEXT: { MyTarget::PtrRC, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { MyTarget::PtrRC, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO-NEXT: { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO-NEXT: { MyTarget::YRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { MyTarget::YRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },

// INSTRINFO-LABEL: extern const int16_t MyTargetRegClassByHwModeTables[4][2] = {
// INSTRINFO-NEXT:   { // DefaultMode
// INSTRINFO-NEXT:     MyTarget::EvenXRegsRegClassID,
// INSTRINFO-NEXT:     MyTarget::XRegsRegClassID,
// INSTRINFO-NEXT:   },
// INSTRINFO-NEXT:   { // XPtr64
// INSTRINFO-NEXT:     MyTarget::EvenXRegsRegClassID,
// INSTRINFO-NEXT:     MyTarget::XRegsRegClassID,
// INSTRINFO-NEXT:   },
// INSTRINFO-NEXT:   { // YPtr32
// INSTRINFO-NEXT:     MyTarget::EvenYRegsRegClassID,
// INSTRINFO-NEXT:     MyTarget::YRegsRegClassID,
// INSTRINFO-NEXT:   },
// INSTRINFO-NEXT:   { // YPtr64
// INSTRINFO-NEXT:     MyTarget::EvenYRegsRegClassID,
// INSTRINFO-NEXT:     MyTarget::YRegsRegClassID,
// INSTRINFO-NEXT:   },
// INSTRINFO-NEXT: };

// ASMWRITER-LABEL: static const AliasPatternCond Conds[] = {
// ASMWRITER-NEXT:    // (TEST_PTRREG PtrRC:$dst, PtrRC:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::PtrRC},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::PtrRC},
// ASMWRITER-NEXT:    // (TEST_PTRREG EvenPtrRC:$dst, EvenPtrRC:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// ASMWRITER-NEXT:    // (TEST_PTRREG NullReg, PtrRC:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Custom, 1/*NullReg*/},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::PtrRC},
// ASMWRITER-NEXT:    // (TEST_PTRREG NullReg, EvenPtrRC:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Custom, 1/*NullReg*/},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// ASMWRITER-NEXT:    // (TEST_PTRREG PtrRegFor64BitModesOnly, PtrRC:$src) - 8
// ASMWRITER-NEXT:    {AliasPatternCond::K_Custom, 2/*PtrRegFor64BitModesOnly*/},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::PtrRC},
// ASMWRITER-NEXT:    // (TEST_XREG X0, XRegs:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Reg, MyTarget::X0},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::XRegsRegClassID},
// ASMWRITER-NEXT:    // (TEST_XREG X0, EvenXRegs:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Reg, MyTarget::X0},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenXRegsRegClassID},
// ASMWRITER-NEXT:    // (TEST_XREG ModeCountReg, XRegs:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Custom, 3/*ModeCountReg*/},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::XRegsRegClassID},
// ASMWRITER-NEXT:    // (TEST_YREG Y0, YRegs:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Reg, MyTarget::Y0},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::YRegsRegClassID},
// ASMWRITER-NEXT:    // (TEST_YREG Y0, EvenYRegs:$src) -
// ASMWRITER-NEXT:    {AliasPatternCond::K_Reg, MyTarget::Y0},
// ASMWRITER-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenYRegsRegClassID},
// ASMWRITER-NEXT:  };

// ASMWRITER-LABEL: static bool MyTargetInstPrinterValidateMCOperand(const MCOperand &MCOp,
// ASMWRITER-NEXT:                    const MCSubtargetInfo &STI,
// ASMWRITER-NEXT:                    unsigned PredicateIndex) {
// ASMWRITER-NEXT:    switch (PredicateIndex) {
// ASMWRITER-NEXT:    default:
// ASMWRITER-NEXT:      llvm_unreachable("Unknown MCOperandPredicate kind");
// ASMWRITER-NEXT:      break;
// ASMWRITER-NEXT:    case 1: {
// ASMWRITER-NEXT:      return MCOp.isReg() && MCOp.getReg() == MyTarget::RegisterByHwMode::getNullReg(STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo));
// ASMWRITER-NEXT:    }
// ASMWRITER-NEXT:    case 2: {
// ASMWRITER-NEXT:      return MCOp.isReg() && MCOp.getReg() == MyTarget::RegisterByHwMode::getPtrRegFor64BitModesOnly(STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo));
// ASMWRITER-NEXT:    }
// ASMWRITER-NEXT:    case 3: {
// ASMWRITER-NEXT:      return MCOp.isReg() && MCOp.getReg() == MyTarget::RegisterByHwMode::getModeCountReg(STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo));
// ASMWRITER-NEXT:    }
// ASMWRITER-NEXT:    }
// ASMWRITER-NEXT:  }



// ASMMATCHER-LABEL: enum InstructionConversionKind {
// ASMMATCHER-NEXT:    Convert__regModeCountReg__Reg1_0,
// ASMMATCHER-NEXT:    Convert__regNullReg__RegByHwMode_PtrRC1_0,
// ASMMATCHER-NEXT:    Convert__regNullReg__RegByHwMode_EvenPtrRC1_0,
// ASMMATCHER-NEXT:    Convert__regX0__Reg1_0,
// ASMMATCHER-NEXT:    Convert__regY0__Reg1_0,
// ASMMATCHER-NEXT:    Convert__regPtrRegFor64BitModesOnly__RegByHwMode_PtrRC1_0,
// ASMMATCHER-NEXT:    Convert__RegByHwMode_PtrRC1_0__RegByHwMode_PtrRC1_1,
// ASMMATCHER-NEXT:    Convert__RegByHwMode_EvenPtrRC1_0__RegByHwMode_EvenPtrRC1_1,
// ASMMATCHER-NEXT:    Convert__Reg1_0__Reg1_1,
// ASMMATCHER-NEXT:    CVT_NUM_SIGNATURES
// ASMMATCHER-NEXT:  };

// ASMMATCHER-LABEL: static const uint8_t ConversionTable[CVT_NUM_SIGNATURES][5] = {
// ASMMATCHER-NEXT:    // Convert__regModeCountReg__Reg1_0
// ASMMATCHER-NEXT:    { CVT_regModeCountReg, 0, CVT_95_Reg, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__regNullReg__RegByHwMode_PtrRC1_0
// ASMMATCHER-NEXT:    { CVT_regNullReg, 0, CVT_95_addRegOperands, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__regNullReg__RegByHwMode_EvenPtrRC1_0
// ASMMATCHER-NEXT:    { CVT_regNullReg, 0, CVT_95_addRegOperands, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__regX0__Reg1_0
// ASMMATCHER-NEXT:    { CVT_regX0, 0, CVT_95_Reg, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__regY0__Reg1_0
// ASMMATCHER-NEXT:    { CVT_regY0, 0, CVT_95_Reg, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__regPtrRegFor64BitModesOnly__RegByHwMode_PtrRC1_0
// ASMMATCHER-NEXT:    { CVT_regPtrRegFor64BitModesOnly, 0, CVT_95_addRegOperands, 1, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__RegByHwMode_PtrRC1_0__RegByHwMode_PtrRC1_1
// ASMMATCHER-NEXT:    { CVT_95_addRegOperands, 1, CVT_95_addRegOperands, 2, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__RegByHwMode_EvenPtrRC1_0__RegByHwMode_EvenPtrRC1_1
// ASMMATCHER-NEXT:    { CVT_95_addRegOperands, 1, CVT_95_addRegOperands, 2, CVT_Done },
// ASMMATCHER-NEXT:    // Convert__Reg1_0__Reg1_1
// ASMMATCHER-NEXT:    { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_Done },
// ASMMATCHER-NEXT:  };

// ASMMATCHER-LABEL: convertToMCInst(unsigned Kind, MCInst &Inst, unsigned Opcode,
// ASMMATCHER-LABEL:  case CVT_regModeCountReg:
// ASMMATCHER-NEXT:     Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getModeCountReg(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_95_Reg:
// ASMMATCHER-NEXT:     static_cast<MyTargetOperand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_regNullReg:
// ASMMATCHER-NEXT:     Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getNullReg(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_95_addRegOperands:
// ASMMATCHER-NEXT:     static_cast<MyTargetOperand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_regX0:
// ASMMATCHER-NEXT:     Inst.addOperand(MCOperand::createReg(MyTarget::X0));
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_regY0:
// ASMMATCHER-NEXT:     Inst.addOperand(MCOperand::createReg(MyTarget::Y0));
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   case CVT_regPtrRegFor64BitModesOnly:
// ASMMATCHER-NEXT:     Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getPtrRegFor64BitModesOnly(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// ASMMATCHER-NEXT:     break;
// ASMMATCHER-NEXT:   }

// ASMMATCHER-LABEL: enum MatchClassKind {
// ASMMATCHER-NEXT:    InvalidMatchClass = 0,
// ASMMATCHER-NEXT:    OptionalMatchClass = 1,
// ASMMATCHER-NEXT:    MCK_LAST_TOKEN = OptionalMatchClass,
// ASMMATCHER-NEXT:    MCK_EvenXRegs, // register class 'EvenXRegs'
// ASMMATCHER-NEXT:    MCK_EvenYRegs, // register class 'EvenYRegs'
// ASMMATCHER-NEXT:    MCK_XRegs, // register class 'XRegs'
// ASMMATCHER-NEXT:    MCK_YRegs, // register class 'YRegs'
// ASMMATCHER-NEXT:    MCK_LAST_REGISTER = MCK_YRegs,
// ASMMATCHER-NEXT:    MCK_RegByHwMode_EvenPtrRC, // register class by hwmode
// ASMMATCHER-NEXT:    MCK_RegByHwMode_PtrRC, // register class by hwmode
// ASMMATCHER-NEXT:    MCK_LAST_REGCLASS_BY_HWMODE = MCK_RegByHwMode_PtrRC,
// ASMMATCHER-NEXT:    MCK_Imm, // user defined class 'ImmAsmOperand'
// ASMMATCHER-NEXT:    NumMatchClassKinds
// ASMMATCHER-NEXT:  };


// ASMMATCHER-LABEL: static const MatchEntry MatchTable0[] = {
// ASMMATCHER-NEXT:    /* mode_count */, MyTarget::TEST_XREG, Convert__regModeCountReg__Reg1_0, AMFBS_None, { MCK_XRegs }, },
// ASMMATCHER-NEXT:    /* t_ptr */, MyTarget::TEST_PTRREG, Convert__regNullReg__RegByHwMode_PtrRC1_0, AMFBS_None, { MCK_RegByHwMode_PtrRC }, },
// ASMMATCHER-NEXT:    /* t_ptr.even */, MyTarget::TEST_PTRREG, Convert__regNullReg__RegByHwMode_EvenPtrRC1_0, AMFBS_None, { MCK_RegByHwMode_EvenPtrRC }, },
// ASMMATCHER-NEXT:    /* t_x */, MyTarget::TEST_XREG, Convert__regX0__Reg1_0, AMFBS_None, { MCK_XRegs }, },
// ASMMATCHER-NEXT:    /* t_x.even */, MyTarget::TEST_XREG, Convert__regX0__Reg1_0, AMFBS_None, { MCK_EvenXRegs }, },
// ASMMATCHER-NEXT:    /* t_y */, MyTarget::TEST_YREG, Convert__regY0__Reg1_0, AMFBS_None, { MCK_YRegs }, },
// ASMMATCHER-NEXT:    /* t_y.even */, MyTarget::TEST_YREG, Convert__regY0__Reg1_0, AMFBS_None, { MCK_EvenYRegs }, },
// ASMMATCHER-NEXT:    /* test_64_only */, MyTarget::TEST_PTRREG, Convert__regPtrRegFor64BitModesOnly__RegByHwMode_PtrRC1_0, AMFBS_None, { MCK_RegByHwMode_PtrRC }, },
// ASMMATCHER-NEXT:    /* test_alias */, MyTarget::TEST_PTRREG, Convert__RegByHwMode_PtrRC1_0__RegByHwMode_PtrRC1_1, AMFBS_None, { MCK_RegByHwMode_PtrRC, MCK_RegByHwMode_PtrRC }, },
// ASMMATCHER-NEXT:    /* test_alias.even */, MyTarget::TEST_PTRREG, Convert__RegByHwMode_EvenPtrRC1_0__RegByHwMode_EvenPtrRC1_1, AMFBS_None, { MCK_RegByHwMode_EvenPtrRC, MCK_RegByHwMode_EvenPtrRC }, },
// ASMMATCHER-NEXT:    /* test_ptr */, MyTarget::TEST_PTRREG, Convert__RegByHwMode_PtrRC1_0__RegByHwMode_PtrRC1_1, AMFBS_None, { MCK_RegByHwMode_PtrRC, MCK_RegByHwMode_PtrRC }, },
// ASMMATCHER-NEXT:    /* test_x */, MyTarget::TEST_XREG, Convert__Reg1_0__Reg1_1, AMFBS_None, { MCK_XRegs, MCK_XRegs }, },
// ASMMATCHER-NEXT:    /* test_y */, MyTarget::TEST_YREG, Convert__Reg1_0__Reg1_1, AMFBS_None, { MCK_YRegs, MCK_YRegs }, },
// ASMMATCHER-NEXT: };

include "Common/RegisterByHwModeCommon.td"

// Define more restrictive subset classes to check that those are handled.
def EvenXRegs : RegisterClass<"MyTarget", [XLenVT], 32, (add X0, X2)> {
  let RegInfos = XLenRI;  // Needed to determine size of registers
}
def EvenYRegs : RegisterClass<"MyTarget", [YLenVT], 64, (add Y0, Y2)> {
  let RegInfos = YLenRI;  // Needed to determine size of registers
}
def EvenPtrRC : RegClassByHwMode<[XPtr32,    XPtr64,    YPtr32,    YPtr64],
                                 [EvenXRegs, EvenXRegs, EvenYRegs, EvenYRegs]>;

def MY_TEST_ALIAS : InstAlias<"test_alias $dst, $src", (TEST_PTRREG PtrRC:$dst, PtrRC:$src)>;
def MY_TEST_ALIAS_EVEN : InstAlias<"test_alias.even $dst, $src", (TEST_PTRREG EvenPtrRC:$dst, EvenPtrRC:$src)>;
def MY_T_X : InstAlias<"t_x $src", (TEST_XREG X0, XRegs:$src)>;
def MY_T_X_EVEN : InstAlias<"t_x.even $src", (TEST_XREG X0, EvenXRegs:$src)>;
def MY_T_Y : InstAlias<"t_y $src", (TEST_YREG Y0, YRegs:$src)>;
def MY_T_Y_EVEN : InstAlias<"t_y.even $src", (TEST_YREG Y0, EvenYRegs:$src)>;
def MY_T_PTR : InstAlias<"t_ptr $src", (TEST_PTRREG NullReg, PtrRC:$src)>;
def MY_T_PTR_EVEN : InstAlias<"t_ptr.even $src", (TEST_PTRREG NullReg, EvenPtrRC:$src)>;

/// Add another test where the register number varies, but the regclass doesn't
def ModeCountReg : RegisterByHwMode<XRegs, [XPtr32, XPtr64, YPtr32, YPtr64],
                                           [X0,     X1,     X2,     X3]>;
def TEST_MODE_COUNT : InstAlias<"mode_count $src", (TEST_XREG ModeCountReg, XRegs:$src)>;

/// Test where we don't have valid values for RegisterByHwMode cases:
def PtrRegFor64BitModesOnly : RegisterByHwMode<PtrRC, [XPtr64, YPtr64], [X1, Y1]>;
let Predicates = [Is64Bit] in
def TEST_64_ONLY : InstAlias<"test_64_only $src", (TEST_PTRREG PtrRegFor64BitModesOnly, PtrRC:$src)>;

/// Include a test for -gen-pseudo-lowering:
class Pseudo<dag outs, dag ins> : TestInstruction {
  let OutOperandList = outs;
  let InOperandList = ins;
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}
// PSEUDO-LABEL: lowerPseudoInstExpansion(const MachineInstr *MI, MCInst &Inst)
// PSEUDO-NEXT: Inst.clear();
// PSEUDO-NEXT: switch (MI->getOpcode()) {
// PSEUDO-NEXT: default: return false;
// PSEUDO-NEXT: case MyTarget::Pseudo64Only: {
// PSEUDO-NEXT:   MCOperand MCOp;
// PSEUDO-NEXT:   Inst.setOpcode(MyTarget::TEST_PTRREG);
// PSEUDO-NEXT:   // Operand: dst
// PSEUDO-NEXT:   Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getPtrRegFor64BitModesOnly(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// PSEUDO-NEXT:   // Operand: src
// PSEUDO-NEXT:   lowerOperand(MI->getOperand(0), MCOp);
// PSEUDO-NEXT:   Inst.addOperand(MCOp);
// PSEUDO-NEXT:   break;
// PSEUDO-NEXT: }
// PSEUDO-NEXT: case MyTarget::PseudoPtr: {
// PSEUDO-NEXT:    MCOperand MCOp;
// PSEUDO-NEXT:    Inst.setOpcode(MyTarget::TEST_PTRREG);
// PSEUDO-NEXT:    // Operand: dst
// PSEUDO-NEXT:    Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getNullReg(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// PSEUDO-NEXT:    // Operand: src
// PSEUDO-NEXT:    lowerOperand(MI->getOperand(0), MCOp);
// PSEUDO-NEXT:    Inst.addOperand(MCOp);
// PSEUDO-NEXT:    break;
// PSEUDO-NEXT:  }
// PSEUDO-NEXT:  case MyTarget::PseudoX: {
// PSEUDO-NEXT:    MCOperand MCOp;
// PSEUDO-NEXT:    Inst.setOpcode(MyTarget::TEST_XREG);
// PSEUDO-NEXT:    // Operand: dst
// PSEUDO-NEXT:    Inst.addOperand(MCOperand::createReg(MyTarget::RegisterByHwMode::getModeCountReg(STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo))));
// PSEUDO-NEXT:    // Operand: src
// PSEUDO-NEXT:    lowerOperand(MI->getOperand(0), MCOp);
// PSEUDO-NEXT:    Inst.addOperand(MCOp);
// PSEUDO-NEXT:    break;
// PSEUDO-NEXT:  }

def PseudoX : Pseudo<(outs), (ins EvenXRegs:$rs1)>,
    PseudoInstExpansion<(TEST_XREG ModeCountReg, XRegs:$rs1)>;
def PseudoPtr : Pseudo<(outs), (ins EvenPtrRC:$rs1)>,
    PseudoInstExpansion<(TEST_PTRREG NullReg, PtrRegOperand:$rs1)>;
let Predicates = [Is64Bit] in
def Pseudo64Only : Pseudo<(outs), (ins PtrRC:$rs1)>,
    PseudoInstExpansion<(TEST_PTRREG PtrRegFor64BitModesOnly, PtrRegOperand:$rs1)>;

// TODO: Add DAGISel support
def int_with_mode_reg : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrWillReturn]>;
def int_with_null_reg : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrWillReturn]>;
def : Pat<(int_with_mode_reg), (TEST_XREG ModeCountReg)>;
def : Pat<(int_with_mode_reg), (TEST_PTRREG NullReg)>;

defm : RemapAllTargetPseudoPointerOperands<PtrRC>;
def MyTargetISA : InstrInfo;
def MyTargetAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
  let AssemblyWriters = [MyTargetAsmWriter];
}
