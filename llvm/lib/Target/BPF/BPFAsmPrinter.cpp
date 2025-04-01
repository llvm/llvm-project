//===-- BPFAsmPrinter.cpp - BPF LLVM assembly writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the BPF assembly language.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFMCInstLower.h"
#include "BTFDebug.h"
#include "MCTargetDesc/BPFInstPrinter.h"
#include "TargetInfo/BPFTargetInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class BPFAsmPrinter : public AsmPrinter {
public:
  explicit BPFAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID), BTF(nullptr) {}

  StringRef getPassName() const override { return "BPF Assembly Printer"; }
  bool doInitialization(Module &M) override;
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             const char *ExtraCode, raw_ostream &O) override;

  void emitInstruction(const MachineInstr *MI) override;
  virtual MCSymbol *GetJTISymbol(unsigned JTID,
                                 bool isLinkerPrivate = false) const override;
  virtual void emitJumpTableInfo() override;

  static char ID;

private:
  BTFDebug *BTF;
};
} // namespace

bool BPFAsmPrinter::doInitialization(Module &M) {
  AsmPrinter::doInitialization(M);

  // Only emit BTF when debuginfo available.
  if (MAI->doesSupportDebugInformation() && !M.debug_compile_units().empty()) {
    BTF = new BTFDebug(this);
    Handlers.push_back(std::unique_ptr<BTFDebug>(BTF));
  }

  return false;
}

void BPFAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << BPFInstPrinter::getRegisterName(MO.getReg());
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

bool BPFAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O);

  printOperand(MI, OpNo, O);
  return false;
}

bool BPFAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, const char *ExtraCode,
                                          raw_ostream &O) {
  assert(OpNum + 1 < MI->getNumOperands() && "Insufficient operands");
  const MachineOperand &BaseMO = MI->getOperand(OpNum);
  const MachineOperand &OffsetMO = MI->getOperand(OpNum + 1);
  assert(BaseMO.isReg() && "Unexpected base pointer for inline asm memory operand.");
  assert(OffsetMO.isImm() && "Unexpected offset for inline asm memory operand.");
  int Offset = OffsetMO.getImm();

  if (ExtraCode)
    return true; // Unknown modifier.

  if (Offset < 0)
    O << "(" << BPFInstPrinter::getRegisterName(BaseMO.getReg()) << " - " << -Offset << ")";
  else
    O << "(" << BPFInstPrinter::getRegisterName(BaseMO.getReg()) << " + " << Offset << ")";

  return false;
}

void BPFAsmPrinter::emitInstruction(const MachineInstr *MI) {
  BPF_MC::verifyInstructionPredicates(MI->getOpcode(),
                                      getSubtargetInfo().getFeatureBits());

  MCInst TmpInst;

  if (!BTF || !BTF->InstLower(MI, TmpInst)) {
    BPFMCInstLower MCInstLowering(OutContext, *this);
    MCInstLowering.Lower(MI, TmpInst);
  }
  EmitToStreamer(*OutStreamer, TmpInst);
}

MCSymbol *BPFAsmPrinter::GetJTISymbol(unsigned JTID,
                                      bool isLinkerPrivate) const {
  SmallString<60> Name;
  raw_svector_ostream(Name)
      << "BPF.JT." << MF->getFunctionNumber() << '.' << JTID;
  MCSymbol *S = OutContext.getOrCreateSymbol(Name);
  if (auto *ES = dyn_cast<MCSymbolELF>(S))
    ES->setBinding(ELF::STB_GLOBAL);
  return S;
}

void BPFAsmPrinter::emitJumpTableInfo() {
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  if (!MJTI)
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty())
    return;

  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  const Function &F = MF->getFunction();
  MCSection *JTS = TLOF.getSectionForJumpTable(F, TM);
  assert(MJTI->getEntryKind() == MachineJumpTableInfo::EK_LabelDifference32);
  unsigned EntrySize = MJTI->getEntrySize(getDataLayout());
  OutStreamer->switchSection(JTS);
  for (unsigned JTI = 0; JTI < JT.size(); JTI++) {
    ArrayRef<MachineBasicBlock *> JTBBs = JT[JTI].MBBs;
    if (JTBBs.empty())
      continue;

    SmallPtrSet<const MachineBasicBlock *, 16> EmittedSets;
    const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
    const MCExpr *Base = TLI->getPICJumpTableRelocBaseExpr(MF, JTI, OutContext);
    for (const MachineBasicBlock *MBB : JTBBs) {
      if (!EmittedSets.insert(MBB).second)
        continue;

      // Offset from gotox to target basic block expressed in number
      // of instructions, e.g.:
      //
      //   .L0_0_set_4 = ((LBB0_4 - .LBPF.JX.0.0) >> 3) - 1
      const MCExpr *LHS = MCSymbolRefExpr::create(MBB->getSymbol(), OutContext);
      OutStreamer->emitAssignment(
          GetJTSetSymbol(JTI, MBB->getNumber()),
          MCBinaryExpr::createSub(
              MCBinaryExpr::createAShr(
                  MCBinaryExpr::createSub(LHS, Base, OutContext),
                  MCConstantExpr::create(3, OutContext), OutContext),
              MCConstantExpr::create(1, OutContext), OutContext));
    }
    // BPF.JT.0.0:
    //    .long   .L0_0_set_4
    //    .long   .L0_0_set_2
    //    ...
    //    .size   BPF.JT.0.0, 128
    MCSymbol *JTStart = GetJTISymbol(JTI);
    OutStreamer->emitLabel(JTStart);
    for (const MachineBasicBlock *MBB : JTBBs) {
      MCSymbol *SetSymbol = GetJTSetSymbol(JTI, MBB->getNumber());
      const MCExpr *V = MCSymbolRefExpr::create(SetSymbol, OutContext);
      OutStreamer->emitValue(V, EntrySize);
    }
    const MCExpr *JTSize = MCConstantExpr::create(JTBBs.size() * 4, OutContext);
    OutStreamer->emitELFSize(JTStart, JTSize);
  }
}

char BPFAsmPrinter::ID = 0;

INITIALIZE_PASS(BPFAsmPrinter, "bpf-asm-printer", "BPF Assembly Printer", false,
                false)

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeBPFAsmPrinter() {
  RegisterAsmPrinter<BPFAsmPrinter> X(getTheBPFleTarget());
  RegisterAsmPrinter<BPFAsmPrinter> Y(getTheBPFbeTarget());
  RegisterAsmPrinter<BPFAsmPrinter> Z(getTheBPFTarget());
}
