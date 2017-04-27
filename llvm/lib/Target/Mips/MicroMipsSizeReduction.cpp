//=== MicroMipsSizeReduction.cpp - MicroMips size reduction pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///\file
/// This pass is used to reduce the size of instructions where applicable.
///
/// TODO: Implement microMIPS64 support.
/// TODO: Implement support for reducing into lwp/swp instruction.
//===----------------------------------------------------------------------===//
#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "micromips-reduce-size"

STATISTIC(NumReduced, "Number of 32-bit instructions reduced to 16-bit ones");

namespace {

/// Order of operands to transfer
// TODO: Will be extended when additional optimizations are added
enum OperandTransfer {
  OT_NA,          ///< Not applicable
  OT_OperandsAll, ///< Transfer all operands
};

/// Reduction type
// TODO: Will be extended when additional optimizations are added
enum ReduceType {
  RT_OneInstr ///< Reduce one instruction into a smaller instruction
};

// Information about immediate field restrictions
struct ImmField {
  ImmField() : ImmFieldOperand(-1), Shift(0), LBound(0), HBound(0) {}
  ImmField(uint8_t Shift, int16_t LBound, int16_t HBound,
           int8_t ImmFieldOperand)
      : ImmFieldOperand(ImmFieldOperand), Shift(Shift), LBound(LBound),
        HBound(HBound) {}
  int8_t ImmFieldOperand; // Immediate operand, -1 if it does not exist
  uint8_t Shift;          // Shift value
  int16_t LBound;         // Low bound of the immediate operand
  int16_t HBound;         // High bound of the immediate operand
};

/// Information about operands
// TODO: Will be extended when additional optimizations are added
struct OpInfo {
  OpInfo(enum OperandTransfer TransferOperands)
      : TransferOperands(TransferOperands) {}
  OpInfo() : TransferOperands(OT_NA) {}

  enum OperandTransfer
      TransferOperands; ///< Operands to transfer to the new instruction
};

// Information about opcodes
struct OpCodes {
  OpCodes(unsigned WideOpc, unsigned NarrowOpc)
      : WideOpc(WideOpc), NarrowOpc(NarrowOpc) {}

  unsigned WideOpc;   ///< Wide opcode
  unsigned NarrowOpc; ///< Narrow opcode
};

/// ReduceTable - A static table with information on mapping from wide
/// opcodes to narrow
struct ReduceEntry {

  enum ReduceType eRType; ///< Reduction type
  bool (*ReduceFunction)(
      MachineInstr *MI,
      const ReduceEntry &Entry); ///< Pointer to reduce function
  struct OpCodes Ops;            ///< All relevant OpCodes
  struct OpInfo OpInf;           ///< Characteristics of operands
  struct ImmField Imm;           ///< Characteristics of immediate field

  ReduceEntry(enum ReduceType RType, struct OpCodes Op,
              bool (*F)(MachineInstr *MI, const ReduceEntry &Entry),
              struct OpInfo OpInf, struct ImmField Imm)
      : eRType(RType), ReduceFunction(F), Ops(Op), OpInf(OpInf), Imm(Imm) {}

  unsigned NarrowOpc() const { return Ops.NarrowOpc; }
  unsigned WideOpc() const { return Ops.WideOpc; }
  int16_t LBound() const { return Imm.LBound; }
  int16_t HBound() const { return Imm.HBound; }
  uint8_t Shift() const { return Imm.Shift; }
  int8_t ImmField() const { return Imm.ImmFieldOperand; }
  enum OperandTransfer TransferOperands() const {
    return OpInf.TransferOperands;
  }
  enum ReduceType RType() const { return eRType; }

  // operator used by std::equal_range
  bool operator<(const unsigned int r) const { return (WideOpc() < r); }

  // operator used by std::equal_range
  friend bool operator<(const unsigned int r, const struct ReduceEntry &re) {
    return (r < re.WideOpc());
  }
};

class MicroMipsSizeReduce : public MachineFunctionPass {
public:
  static char ID;
  MicroMipsSizeReduce();

  static const MipsInstrInfo *MipsII;
  const MipsSubtarget *Subtarget;

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "microMIPS instruction size reduction pass";
  }

private:
  /// Reduces width of instructions in the specified basic block.
  bool ReduceMBB(MachineBasicBlock &MBB);

  /// Attempts to reduce MI, returns true on success.
  bool ReduceMI(const MachineBasicBlock::instr_iterator &MII);

  // Attempts to reduce LW/SW instruction into LWSP/SWSP,
  // returns true on success.
  static bool ReduceXWtoXWSP(MachineInstr *MI, const ReduceEntry &Entry);

  // Attempts to reduce arithmetic instructions, returns true on success
  static bool ReduceArithmeticInstructions(MachineInstr *MI,
                                           const ReduceEntry &Entry);

  // Changes opcode of an instruction
  static bool ReplaceInstruction(MachineInstr *MI, const ReduceEntry &Entry);

  // Table with transformation rules for each instruction
  static llvm::SmallVector<ReduceEntry, 16> ReduceTable;
};

char MicroMipsSizeReduce::ID = 0;
const MipsInstrInfo *MicroMipsSizeReduce::MipsII;

// This table must be sorted by WideOpc as a main criterion and
// ReduceType as a sub-criterion (when wide opcodes are the same)
llvm::SmallVector<ReduceEntry, 16> MicroMipsSizeReduce::ReduceTable = {

    // ReduceType, OpCodes, ReduceFunction,
    // OpInfo(TransferOperands),
    // ImmField(Shift, LBound, HBound, ImmFieldPosition)
    {RT_OneInstr, OpCodes(Mips::ADDu, Mips::ADDU16_MM),
     ReduceArithmeticInstructions, OpInfo(OT_OperandsAll),
     ImmField(0, 0, 0, -1)},
    {RT_OneInstr, OpCodes(Mips::ADDu_MM, Mips::ADDU16_MM),
     ReduceArithmeticInstructions, OpInfo(OT_OperandsAll),
     ImmField(0, 0, 0, -1)},
    {RT_OneInstr, OpCodes(Mips::LW, Mips::LWSP_MM), ReduceXWtoXWSP,
     OpInfo(OT_OperandsAll), ImmField(2, 0, 32, 2)},
    {RT_OneInstr, OpCodes(Mips::LW_MM, Mips::LWSP_MM), ReduceXWtoXWSP,
     OpInfo(OT_OperandsAll), ImmField(2, 0, 32, 2)},
    {RT_OneInstr, OpCodes(Mips::SUBu, Mips::SUBU16_MM),
     ReduceArithmeticInstructions, OpInfo(OT_OperandsAll),
     ImmField(0, 0, 0, -1)},
    {RT_OneInstr, OpCodes(Mips::SUBu_MM, Mips::SUBU16_MM),
     ReduceArithmeticInstructions, OpInfo(OT_OperandsAll),
     ImmField(0, 0, 0, -1)},
    {RT_OneInstr, OpCodes(Mips::SW, Mips::SWSP_MM), ReduceXWtoXWSP,
     OpInfo(OT_OperandsAll), ImmField(2, 0, 32, 2)},
    {RT_OneInstr, OpCodes(Mips::SW_MM, Mips::SWSP_MM), ReduceXWtoXWSP,
     OpInfo(OT_OperandsAll), ImmField(2, 0, 32, 2)},
};
}

// Returns true if the machine operand MO is register SP
static bool IsSP(const MachineOperand &MO) {
  if (MO.isReg() && ((MO.getReg() == Mips::SP)))
    return true;
  return false;
}

// Returns true if the machine operand MO is register $16, $17, or $2-$7.
static bool isMMThreeBitGPRegister(const MachineOperand &MO) {
  if (MO.isReg() && Mips::GPRMM16RegClass.contains(MO.getReg()))
    return true;
  return false;
}

// Returns true if the operand Op is an immediate value
// and writes the immediate value into variable Imm
static bool GetImm(MachineInstr *MI, unsigned Op, int64_t &Imm) {

  if (!MI->getOperand(Op).isImm())
    return false;
  Imm = MI->getOperand(Op).getImm();
  return true;
}

// Returns true if the variable Value has the number of least-significant zero
// bits equal to Shift and if the shifted value is between the bounds
static bool InRange(int64_t Value, unsigned short Shift, int LBound,
                    int HBound) {
  int64_t Value2 = Value >> Shift;
  if ((Value2 << Shift) == Value && (Value2 >= LBound) && (Value2 < HBound))
    return true;
  return false;
}

// Returns true if immediate operand is in range
static bool ImmInRange(MachineInstr *MI, const ReduceEntry &Entry) {

  int64_t offset;

  if (!GetImm(MI, Entry.ImmField(), offset))
    return false;

  if (!InRange(offset, Entry.Shift(), Entry.LBound(), Entry.HBound()))
    return false;

  return true;
}

MicroMipsSizeReduce::MicroMipsSizeReduce() : MachineFunctionPass(ID) {}

bool MicroMipsSizeReduce::ReduceMI(
    const MachineBasicBlock::instr_iterator &MII) {

  MachineInstr *MI = &*MII;
  unsigned Opcode = MI->getOpcode();

  // Search the table.
  llvm::SmallVector<ReduceEntry, 16>::const_iterator Start =
      std::begin(ReduceTable);
  llvm::SmallVector<ReduceEntry, 16>::const_iterator End =
      std::end(ReduceTable);

  std::pair<llvm::SmallVector<ReduceEntry, 16>::const_iterator,
            llvm::SmallVector<ReduceEntry, 16>::const_iterator>
      Range = std::equal_range(Start, End, Opcode);

  if (Range.first == Range.second)
    return false;

  for (llvm::SmallVector<ReduceEntry, 16>::const_iterator Entry = Range.first;
       Entry != Range.second; ++Entry)
    if (((*Entry).ReduceFunction)(&(*MII), *Entry))
      return true;

  return false;
}

bool MicroMipsSizeReduce::ReduceXWtoXWSP(MachineInstr *MI,
                                         const ReduceEntry &Entry) {

  if (!ImmInRange(MI, Entry))
    return false;

  if (!IsSP(MI->getOperand(1)))
    return false;

  return ReplaceInstruction(MI, Entry);
}

bool MicroMipsSizeReduce::ReduceArithmeticInstructions(
    MachineInstr *MI, const ReduceEntry &Entry) {

  if (!isMMThreeBitGPRegister(MI->getOperand(0)) ||
      !isMMThreeBitGPRegister(MI->getOperand(1)) ||
      !isMMThreeBitGPRegister(MI->getOperand(2)))
    return false;

  return ReplaceInstruction(MI, Entry);
}

bool MicroMipsSizeReduce::ReduceMBB(MachineBasicBlock &MBB) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                    E = MBB.instr_end();
  MachineBasicBlock::instr_iterator NextMII;

  // Iterate through the instructions in the basic block
  for (; MII != E; MII = NextMII) {
    NextMII = std::next(MII);
    MachineInstr *MI = &*MII;

    // Don't reduce bundled instructions or pseudo operations
    if (MI->isBundle() || MI->isTransient())
      continue;

    // Try to reduce 32-bit instruction into 16-bit instruction
    Modified |= ReduceMI(MII);
  }

  return Modified;
}

bool MicroMipsSizeReduce::ReplaceInstruction(MachineInstr *MI,
                                             const ReduceEntry &Entry) {

  MI->setDesc(MipsII->get(Entry.NarrowOpc()));
  DEBUG(dbgs() << "Converted into 16-bit: " << *MI);
  ++NumReduced;
  return true;
}

bool MicroMipsSizeReduce::runOnMachineFunction(MachineFunction &MF) {

  Subtarget = &static_cast<const MipsSubtarget &>(MF.getSubtarget());

  // TODO: Add support for other subtargets:
  // microMIPS32r6 and microMIPS64r6
  if (!Subtarget->inMicroMipsMode() || !Subtarget->hasMips32r2())
    return false;

  MipsII = static_cast<const MipsInstrInfo *>(Subtarget->getInstrInfo());

  bool Modified = false;
  MachineFunction::iterator I = MF.begin(), E = MF.end();

  for (; I != E; ++I)
    Modified |= ReduceMBB(*I);
  return Modified;
}

/// Returns an instance of the MicroMips size reduction pass.
FunctionPass *llvm::createMicroMipsSizeReductionPass() {
  return new MicroMipsSizeReduce();
}
