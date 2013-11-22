//==-- AArch64InstPrinter.cpp - Convert AArch64 MCInst to assembly syntax --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an AArch64 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "AArch64InstPrinter.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "AArch64GenAsmWriter.inc"

static int64_t unpackSignedImm(int BitWidth, uint64_t Value) {
  assert(!(Value & ~((1ULL << BitWidth)-1)) && "immediate not n-bit");
  if (Value & (1ULL <<  (BitWidth - 1)))
    return static_cast<int64_t>(Value) - (1LL << BitWidth);
  else
    return Value;
}

AArch64InstPrinter::AArch64InstPrinter(const MCAsmInfo &MAI,
                                       const MCInstrInfo &MII,
                                       const MCRegisterInfo &MRI,
                                       const MCSubtargetInfo &STI) :
  MCInstPrinter(MAI, MII, MRI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

void AArch64InstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void
AArch64InstPrinter::printOffsetSImm9Operand(const MCInst *MI,
                                              unsigned OpNum, raw_ostream &O) {
  const MCOperand &MOImm = MI->getOperand(OpNum);
  int32_t Imm = unpackSignedImm(9, MOImm.getImm());

  O << '#' << Imm;
}

void
AArch64InstPrinter::printAddrRegExtendOperand(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O, unsigned MemSize,
                                          unsigned RmSize) {
  unsigned ExtImm = MI->getOperand(OpNum).getImm();
  unsigned OptionHi = ExtImm >> 1;
  unsigned S = ExtImm & 1;
  bool IsLSL = OptionHi == 1 && RmSize == 64;

  const char *Ext;
  switch (OptionHi) {
  case 1:
    Ext = (RmSize == 32) ? "uxtw" : "lsl";
    break;
  case 3:
    Ext = (RmSize == 32) ? "sxtw" : "sxtx";
    break;
  default:
    llvm_unreachable("Incorrect Option on load/store (reg offset)");
  }
  O << Ext;

  if (S) {
    unsigned ShiftAmt = Log2_32(MemSize);
    O << " #" << ShiftAmt;
  } else if (IsLSL) {
    O << " #0";
  }
}

void
AArch64InstPrinter::printAddSubImmLSL0Operand(const MCInst *MI,
                                              unsigned OpNum, raw_ostream &O) {
  const MCOperand &Imm12Op = MI->getOperand(OpNum);

  if (Imm12Op.isImm()) {
    int64_t Imm12 = Imm12Op.getImm();
    assert(Imm12 >= 0 && "Invalid immediate for add/sub imm");
    O << "#" << Imm12;
  } else {
    assert(Imm12Op.isExpr() && "Unexpected shift operand type");
    O << "#" << *Imm12Op.getExpr();
  }
}

void
AArch64InstPrinter::printAddSubImmLSL12Operand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &O) {

  printAddSubImmLSL0Operand(MI, OpNum, O);

  O << ", lsl #12";
}

void
AArch64InstPrinter::printBareImmOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  O << MO.getImm();
}

template<unsigned RegWidth> void
AArch64InstPrinter::printBFILSBOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &ImmROp = MI->getOperand(OpNum);
  unsigned LSB = ImmROp.getImm() == 0 ? 0 : RegWidth - ImmROp.getImm();

  O << '#' << LSB;
}

void AArch64InstPrinter::printBFIWidthOperand(const MCInst *MI, unsigned OpNum,
                                              raw_ostream &O) {
  const MCOperand &ImmSOp = MI->getOperand(OpNum);
  unsigned Width = ImmSOp.getImm() + 1;

  O << '#' << Width;
}

void
AArch64InstPrinter::printBFXWidthOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &ImmSOp = MI->getOperand(OpNum);
  const MCOperand &ImmROp = MI->getOperand(OpNum - 1);

  unsigned ImmR = ImmROp.getImm();
  unsigned ImmS = ImmSOp.getImm();

  assert(ImmS >= ImmR && "Invalid ImmR, ImmS combination for bitfield extract");

  O << '#' << (ImmS - ImmR + 1);
}

void
AArch64InstPrinter::printCRxOperand(const MCInst *MI, unsigned OpNum,
                                    raw_ostream &O) {
    const MCOperand &CRx = MI->getOperand(OpNum);

    O << 'c' << CRx.getImm();
}


void
AArch64InstPrinter::printCVTFixedPosOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
    const MCOperand &ScaleOp = MI->getOperand(OpNum);

    O << '#' << (64 - ScaleOp.getImm());
}


void AArch64InstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &o) {
  const MCOperand &MOImm8 = MI->getOperand(OpNum);

  assert(MOImm8.isImm()
         && "Immediate operand required for floating-point immediate inst");

  uint32_t Imm8 = MOImm8.getImm();
  uint32_t Fraction = Imm8 & 0xf;
  uint32_t Exponent = (Imm8 >> 4) & 0x7;
  uint32_t Negative = (Imm8 >> 7) & 0x1;

  float Val = 1.0f + Fraction / 16.0f;

  // That is:
  // 000 -> 2^1,  001 -> 2^2,  010 -> 2^3,  011 -> 2^4,
  // 100 -> 2^-3, 101 -> 2^-2, 110 -> 2^-1, 111 -> 2^0
  if (Exponent & 0x4) {
    Val /= 1 << (7 - Exponent);
  } else {
    Val *= 1 << (Exponent + 1);
  }

  Val = Negative ? -Val : Val;

  o << '#' << format("%.8f", Val);
}

void AArch64InstPrinter::printFPZeroOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &o) {
  o << "#0.0";
}

void
AArch64InstPrinter::printCondCodeOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  O << A64CondCodeToString(static_cast<A64CC::CondCodes>(MO.getImm()));
}

template <unsigned field_width, unsigned scale> void
AArch64InstPrinter::printLabelOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  if (!MO.isImm()) {
    printOperand(MI, OpNum, O);
    return;
  }

  // The immediate of LDR (lit) instructions is a signed 19-bit immediate, which
  // is multiplied by 4 (because all A64 instructions are 32-bits wide).
  uint64_t UImm = MO.getImm();
  uint64_t Sign = UImm & (1LL << (field_width - 1));
  int64_t SImm = scale * ((UImm & ~Sign) - Sign);

  O << "#" << SImm;
}

template<unsigned RegWidth> void
AArch64InstPrinter::printLogicalImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  uint64_t Val;
  A64Imms::isLogicalImmBits(RegWidth, MO.getImm(), Val);
  O << "#0x";
  O.write_hex(Val);
}

void
AArch64InstPrinter::printOffsetUImm12Operand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &O, int MemSize) {
  const MCOperand &MOImm = MI->getOperand(OpNum);

  if (MOImm.isImm()) {
    uint32_t Imm = MOImm.getImm() * MemSize;

    O << "#" << Imm;
  } else {
    O << "#" << *MOImm.getExpr();
  }
}

void
AArch64InstPrinter::printShiftOperand(const MCInst *MI,  unsigned OpNum,
                                      raw_ostream &O,
                                      A64SE::ShiftExtSpecifiers Shift) {
    const MCOperand &MO = MI->getOperand(OpNum);

    // LSL #0 is not printed
    if (Shift == A64SE::LSL && MO.isImm() && MO.getImm() == 0)
        return;

    switch (Shift) {
    case A64SE::LSL: O << "lsl"; break;
    case A64SE::LSR: O << "lsr"; break;
    case A64SE::ASR: O << "asr"; break;
    case A64SE::ROR: O << "ror"; break;
    default: llvm_unreachable("Invalid shift specifier in logical instruction");
    }

  O << " #" << MO.getImm();
}

void
AArch64InstPrinter::printMoveWideImmOperand(const MCInst *MI,  unsigned OpNum,
                                            raw_ostream &O) {
  const MCOperand &UImm16MO = MI->getOperand(OpNum);
  const MCOperand &ShiftMO = MI->getOperand(OpNum + 1);

  if (UImm16MO.isImm()) {
    O << '#' << UImm16MO.getImm();

    if (ShiftMO.getImm() != 0)
      O << ", lsl #" << (ShiftMO.getImm() * 16);

    return;
  }

  O << "#" << *UImm16MO.getExpr();
}

void AArch64InstPrinter::printNamedImmOperand(const NamedImmMapper &Mapper,
                                              const MCInst *MI, unsigned OpNum,
                                              raw_ostream &O) {
  bool ValidName;
  const MCOperand &MO = MI->getOperand(OpNum);
  StringRef Name = Mapper.toString(MO.getImm(), ValidName);

  if (ValidName)
    O << Name;
  else
    O << '#' << MO.getImm();
}

void
AArch64InstPrinter::printSysRegOperand(const A64SysReg::SysRegMapper &Mapper,
                                       const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  bool ValidName;
  std::string Name = Mapper.toString(MO.getImm(), ValidName);
  if (ValidName) {
    O << Name;
    return;
  }
}


void AArch64InstPrinter::printRegExtendOperand(const MCInst *MI,
                                               unsigned OpNum,
                                               raw_ostream &O,
                                               A64SE::ShiftExtSpecifiers Ext) {
  // FIXME: In principle TableGen should be able to detect this itself far more
  // easily. We will only accumulate more of these hacks.
  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();

  if (isStackReg(Reg0) || isStackReg(Reg1)) {
    A64SE::ShiftExtSpecifiers LSLEquiv;

    if (Reg0 == AArch64::XSP || Reg1 == AArch64::XSP)
      LSLEquiv = A64SE::UXTX;
    else
      LSLEquiv = A64SE::UXTW;

    if (Ext == LSLEquiv) {
      O << "lsl #" << MI->getOperand(OpNum).getImm();
      return;
    }
  }

  switch (Ext) {
  case A64SE::UXTB: O << "uxtb"; break;
  case A64SE::UXTH: O << "uxth"; break;
  case A64SE::UXTW: O << "uxtw"; break;
  case A64SE::UXTX: O << "uxtx"; break;
  case A64SE::SXTB: O << "sxtb"; break;
  case A64SE::SXTH: O << "sxth"; break;
  case A64SE::SXTW: O << "sxtw"; break;
  case A64SE::SXTX: O << "sxtx"; break;
  default: llvm_unreachable("Unexpected shift type for printing");
  }

  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.getImm() != 0)
    O << " #" << MO.getImm();
}

template<int MemScale> void
AArch64InstPrinter::printSImm7ScaledOperand(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  const MCOperand &MOImm = MI->getOperand(OpNum);
  int32_t Imm = unpackSignedImm(7, MOImm.getImm());

  O << "#" << (Imm * MemScale);
}

void AArch64InstPrinter::printVPRRegister(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  unsigned Reg = MI->getOperand(OpNo).getReg();
  std::string Name = getRegisterName(Reg);
  Name[0] = 'v';
  O << Name;
}

void AArch64InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    O << getRegisterName(Reg);
  } else if (Op.isImm()) {
    O << '#' << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    // If a symbolic branch target was added as a constant expression then print
    // that address in hex.
    const MCConstantExpr *BranchTarget = dyn_cast<MCConstantExpr>(Op.getExpr());
    int64_t Address;
    if (BranchTarget && BranchTarget->EvaluateAsAbsolute(Address)) {
      O << "0x";
      O.write_hex(Address);
    }
    else {
      // Otherwise, just print the expression.
      O << *Op.getExpr();
    }
  }
}


void AArch64InstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                   StringRef Annot) {
  if (MI->getOpcode() == AArch64::TLSDESCCALL) {
    // This is a special assembler directive which applies an
    // R_AARCH64_TLSDESC_CALL to the following (BLR) instruction. It has a fixed
    // form outside the normal TableGenerated scheme.
    O << "\t.tlsdesccall " << *MI->getOperand(0).getExpr();
  } else if (!printAliasInstr(MI, O))
    printInstruction(MI, O);

  printAnnotation(O, Annot);
}

template <A64SE::ShiftExtSpecifiers Ext, bool isHalf>
void AArch64InstPrinter::printNeonMovImmShiftOperand(const MCInst *MI,
                                                     unsigned OpNum,
                                                     raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  assert(MO.isImm() &&
         "Immediate operand required for Neon vector immediate inst.");

  bool IsLSL = false;
  if (Ext == A64SE::LSL)
    IsLSL = true;
  else if (Ext != A64SE::MSL)
    llvm_unreachable("Invalid shift specifier in movi instruction");

  int64_t Imm = MO.getImm();

  // MSL and LSLH accepts encoded shift amount 0 or 1.
  if ((!IsLSL || (IsLSL && isHalf)) && Imm != 0 && Imm != 1)
    llvm_unreachable("Invalid shift amount in movi instruction");

  // LSH accepts encoded shift amount 0, 1, 2 or 3.
  if (IsLSL && (Imm < 0 || Imm > 3))
    llvm_unreachable("Invalid shift amount in movi instruction");

  // Print shift amount as multiple of 8 with MSL encoded shift amount
  // 0 and 1 printed as 8 and 16.
  if (!IsLSL)
    Imm++;
  Imm *= 8;

  // LSL #0 is not printed
  if (IsLSL) {
    if (Imm == 0)
      return;
    O << ", lsl";
  } else
    O << ", msl";

  O << " #" << Imm;
}

void AArch64InstPrinter::printNeonUImm0Operand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &o) {
  o << "#0x0";
}

void AArch64InstPrinter::printUImmHexOperand(const MCInst *MI, unsigned OpNum,
                                             raw_ostream &O) {
  const MCOperand &MOUImm = MI->getOperand(OpNum);

  assert(MOUImm.isImm() &&
         "Immediate operand required for Neon vector immediate inst.");

  unsigned Imm = MOUImm.getImm();

  O << "#0x";
  O.write_hex(Imm);
}

void AArch64InstPrinter::printUImmBareOperand(const MCInst *MI,
                                              unsigned OpNum,
                                              raw_ostream &O) {
  const MCOperand &MOUImm = MI->getOperand(OpNum);

  assert(MOUImm.isImm()
         && "Immediate operand required for Neon vector immediate inst.");

  unsigned Imm = MOUImm.getImm();
  O << Imm;
}

void AArch64InstPrinter::printNeonUImm64MaskOperand(const MCInst *MI,
                                                    unsigned OpNum,
                                                    raw_ostream &O) {
  const MCOperand &MOUImm8 = MI->getOperand(OpNum);

  assert(MOUImm8.isImm() &&
         "Immediate operand required for Neon vector immediate bytemask inst.");

  uint32_t UImm8 = MOUImm8.getImm();
  uint64_t Mask = 0;

  // Replicates 0x00 or 0xff byte in a 64-bit vector
  for (unsigned ByteNum = 0; ByteNum < 8; ++ByteNum) {
    if ((UImm8 >> ByteNum) & 1)
      Mask |= (uint64_t)0xff << (8 * ByteNum);
  }

  O << "#0x";
  O.write_hex(Mask);
}

// If Count > 1, there are two valid kinds of vector list:
//   (1) {Vn.layout, Vn+1.layout, ... , Vm.layout}
//   (2) {Vn.layout - Vm.layout}
// We choose the first kind as output.
template <A64Layout::VectorLayout Layout, unsigned Count>
void AArch64InstPrinter::printVectorList(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  assert(Count >= 1 && Count <= 4 && "Invalid Number of Vectors");

  unsigned Reg = MI->getOperand(OpNum).getReg();
  std::string LayoutStr = A64VectorLayoutToString(Layout);
  O << "{";
  if (Count > 1) { // Print sub registers separately
    bool IsVec64 = (Layout < A64Layout::VL_16B);
    unsigned SubRegIdx = IsVec64 ? AArch64::dsub_0 : AArch64::qsub_0;
    for (unsigned I = 0; I < Count; I++) {
      std::string Name = getRegisterName(MRI.getSubReg(Reg, SubRegIdx++));
      Name[0] = 'v';
      O << Name << LayoutStr;
      if (I != Count - 1)
        O << ", ";
    }
  } else { // Print the register directly when NumVecs is 1.
    std::string Name = getRegisterName(Reg);
    Name[0] = 'v';
    O << Name << LayoutStr;
  }
  O << "}";
}
