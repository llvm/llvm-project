//===-- Next32AsmPrinter.cpp - Next32 LLVM assembly writer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the Next32 assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32InstPrinter.h"
#include "MCTargetDesc/Next32MCExpr.h"
#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32MCInstLower.h"
#include "Next32TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

static cl::opt<bool> EmitNamedBBLabels(
    "next32-emit-named-bb-labels", cl::init(false),
    cl::desc("Next32: Emit labels for BasicBlocks according to their IR name. "
             "Requires that the LLVM context does not discard value names, and "
             "that the names are already globally unique within the module."),
    cl::Hidden);

namespace {
class Next32AsmPrinter : public AsmPrinter {
public:
  explicit Next32AsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "Next32 Assembly Printer"; }
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             const char *ExtraCode, raw_ostream &OS) override;

  const MCExpr *lowerConstant(const Constant *CV) override;
  void emitInstruction(const MachineInstr *MI) override;
  void EmitBasicBlockNameLabel(const MachineBasicBlock &MBB) const;
  void emitBasicBlockStart(const MachineBasicBlock &MBB) override;

  void EmitHelperSymbols(const MachineBasicBlock &MBB);
};
} // namespace

void Next32AsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                    raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << Next32InstPrinter::getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_BlockAddress: {
    MCSymbol *BA = GetBlockAddressSymbol(MO.getBlockAddress());
    O << BA->getName();
    break;
  }

  case MachineOperand::MO_ExternalSymbol:
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    break;

  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  default:
    llvm_unreachable("<unknown operand type>");
  }
}

bool Next32AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       const char *ExtraCode, raw_ostream &OS) {
  if (ExtraCode && ExtraCode[0])
    return true; // Next32 does not have special modifiers

  printOperand(MI, OpNo, OS);
  return false;
}

bool Next32AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                             unsigned OpNum,
                                             const char *ExtraCode,
                                             raw_ostream &OS) {
  assert(OpNum + 1 < MI->getNumOperands() && "Insufficient operands");
  const MachineOperand &BaseMO = MI->getOperand(OpNum);
  const MachineOperand &OffsetMO = MI->getOperand(OpNum + 1);
  assert(BaseMO.isReg() &&
         "Unexpected base pointer for inline asm memory operand.");
  assert(OffsetMO.isImm() &&
         "Unexpected offset for inline asm memory operand.");
  int Offset = OffsetMO.getImm();

  if (ExtraCode)
    return true; // Unknown modifier.

  if (Offset < 0)
    OS << "(" << Next32InstPrinter::getRegisterName(BaseMO.getReg()) << " - "
       << -Offset << ")";
  else
    OS << "(" << Next32InstPrinter::getRegisterName(BaseMO.getReg()) << " + "
       << Offset << ")";

  return false;
}

const MCExpr *Next32AsmPrinter::lowerConstant(const Constant *CV) {
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    const GlobalObject *GO = GV->getAliaseeObject();
    if (const Function *F = dyn_cast<Function>(GO))
      return MCSymbolRefExpr::create(
          getSymbol(F), llvm::MCSymbolRefExpr::VK_Next32_FUNC_PTR, OutContext);
  }

  return AsmPrinter::lowerConstant(CV);
}

void Next32AsmPrinter::emitInstruction(const MachineInstr *MI) {
  Next32_MC::verifyInstructionPredicates(MI->getOpcode(),
                                         getSubtargetInfo().getFeatureBits());

  if (MI->getOpcode() == TargetOpcode::PATCHABLE_FUNCTION_ENTER ||
      MI->getOpcode() == TargetOpcode::PATCHABLE_RET ||
      MI->getOpcode() == TargetOpcode::PATCHABLE_TAIL_CALL)
    return;
  Next32MCInstLower MCInstLowering(OutContext, *this);
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  if (MI->getOpcode() == Next32::SYM_INSTR) {
    OutStreamer->emitLabel(MI->getOperand(0).getMCSymbol());
    return;
  }
  EmitToStreamer(*OutStreamer, TmpInst);
}

void Next32AsmPrinter::EmitBasicBlockNameLabel(
    const MachineBasicBlock &MBB) const {
  if (!EmitNamedBBLabels)
    return;

  const BasicBlock *BB = MBB.getBasicBlock();
  assert(BB && "No BasicBlock for MachineBasicBlock?");
  assert(!BB->getContext().shouldDiscardValueNames() &&
         "Cannot emit named BB labels in name-discarding LLVMContext");

  const bool IsEntryBlock = BB == &BB->getParent()->getEntryBlock();
  if (IsEntryBlock || !BB->hasName())
    return;

  MCContext &Ctx = MBB.getParent()->getContext();
  MCSymbol *Symbol = Ctx.getOrCreateSymbol(BB->getName());
  if (!Symbol->isUndefined())
    // There may be multiple MBBs belonging to the same BB due to CallSplit.
    // Emit the label only for the first one.
    return;

  OutStreamer->emitLabel(Symbol);
}

void Next32AsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  EmitBasicBlockNameLabel(MBB);

  if (MBB.pred_empty() ||
      (isBlockOnlyReachableByFallthrough(&MBB) && !MBB.isEHFuncletEntry())) {
    // Make sure there is a symbol at the beginning of every BB
    // (see implemenation of AsmPrinter::EmitBasicBlockStart)
    OutStreamer->emitLabel(MBB.getSymbol());
  }
  AsmPrinter::emitBasicBlockStart(MBB);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32AsmPrinter() {
  RegisterAsmPrinter<Next32AsmPrinter> X(getTheNext32Target());
}
