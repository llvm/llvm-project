//===-- XCoreAsmPrinter.cpp - XCore LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the XAS-format XCore assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "XCore.h"
#include "XCoreInstrInfo.h"
#include "XCoreSubtarget.h"
#include "XCoreTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
using namespace llvm;

static cl::opt<unsigned> MaxThreads("xcore-max-threads", cl::Optional,
  cl::desc("Maximum number of threads (for emulation thread-local storage)"),
  cl::Hidden,
  cl::value_desc("number"),
  cl::init(8));

namespace {
  class XCoreAsmPrinter : public AsmPrinter {
    const XCoreSubtarget &Subtarget;
    void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);
  public:
    explicit XCoreAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer), Subtarget(TM.getSubtarget<XCoreSubtarget>()){}

    virtual const char *getPassName() const {
      return "XCore Assembly Printer";
    }

    void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
    void printInlineJT(const MachineInstr *MI, int opNum, raw_ostream &O,
                       const std::string &directive = ".jmptable");
    void printInlineJT32(const MachineInstr *MI, int opNum, raw_ostream &O) {
      printInlineJT(MI, opNum, O, ".jmptable32");
    }
    void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);

    void emitArrayBound(MCSymbol *Sym, const GlobalVariable *GV);
    virtual void EmitGlobalVariable(const GlobalVariable *GV);

    void printInstruction(const MachineInstr *MI, raw_ostream &O); // autogen'd.
    static const char *getRegisterName(unsigned RegNo);

    void EmitFunctionEntryLabel();
    void EmitInstruction(const MachineInstr *MI);
    void EmitFunctionBodyEnd();
    virtual MachineLocation getDebugValueLocation(const MachineInstr *MI) const;
  };
} // end of anonymous namespace

#include "XCoreGenAsmWriter.inc"

void XCoreAsmPrinter::emitArrayBound(MCSymbol *Sym, const GlobalVariable *GV) {
  assert(((GV->hasExternalLinkage() ||
    GV->hasWeakLinkage()) ||
    GV->hasLinkOnceLinkage()) && "Unexpected linkage");
  if (ArrayType *ATy = dyn_cast<ArrayType>(
    cast<PointerType>(GV->getType())->getElementType())) {
    OutStreamer.EmitSymbolAttribute(Sym, MCSA_Global);
    // FIXME: MCStreamerize.
    OutStreamer.EmitRawText(StringRef(".globound"));
    OutStreamer.EmitRawText("\t.set\t" + Twine(Sym->getName()));
    OutStreamer.EmitRawText(".globound," + Twine(ATy->getNumElements()));
    if (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage()) {
      // TODO Use COMDAT groups for LinkOnceLinkage
      OutStreamer.EmitRawText(MAI->getWeakDefDirective() +Twine(Sym->getName())+
                              ".globound");
    }
  }
}

void XCoreAsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {
  // Check to see if this is a special global used by LLVM, if so, emit it.
  if (!GV->hasInitializer() ||
      EmitSpecialLLVMGlobal(GV))
    return;

  const DataLayout *TD = TM.getDataLayout();
  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(GV, Mang,TM));

  
  MCSymbol *GVSym = Mang->getSymbol(GV);
  const Constant *C = GV->getInitializer();
  unsigned Align = (unsigned)TD->getPreferredTypeAlignmentShift(C->getType());
  
  // Mark the start of the global
  OutStreamer.EmitRawText("\t.cc_top " + Twine(GVSym->getName()) + ".data," +
                          GVSym->getName());

  switch (GV->getLinkage()) {
  case GlobalValue::AppendingLinkage:
    report_fatal_error("AppendingLinkage is not supported by this target!");
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::ExternalLinkage:
    emitArrayBound(GVSym, GV);
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);

    // TODO Use COMDAT groups for LinkOnceLinkage
    if (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage())
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Weak);
    // FALL THROUGH
  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
    break;
  case GlobalValue::DLLImportLinkage:
    llvm_unreachable("DLLImport linkage is not supported by this target!");
  case GlobalValue::DLLExportLinkage:
    llvm_unreachable("DLLExport linkage is not supported by this target!");
  default:
    llvm_unreachable("Unknown linkage type!");
  }

  EmitAlignment(Align > 2 ? Align : 2, GV);
  
  unsigned Size = TD->getTypeAllocSize(C->getType());
  if (GV->isThreadLocal()) {
    Size *= MaxThreads;
  }
  if (MAI->hasDotTypeDotSizeDirective()) {
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_ELF_TypeObject);
    OutStreamer.EmitRawText("\t.size " + Twine(GVSym->getName()) + "," +
                            Twine(Size));
  }
  OutStreamer.EmitLabel(GVSym);
  
  EmitGlobalConstant(C);
  if (GV->isThreadLocal()) {
    for (unsigned i = 1; i < MaxThreads; ++i)
      EmitGlobalConstant(C);
  }
  // The ABI requires that unsigned scalar types smaller than 32 bits
  // are padded to 32 bits.
  if (Size < 4)
    OutStreamer.EmitZeros(4 - Size, 0);
  
  // Mark the end of the global
  OutStreamer.EmitRawText("\t.cc_bottom " + Twine(GVSym->getName()) + ".data");
}

/// EmitFunctionBodyEnd - Targets can override this to emit stuff after
/// the last basic block in the function.
void XCoreAsmPrinter::EmitFunctionBodyEnd() {
  // Emit function end directives
  OutStreamer.EmitRawText("\t.cc_bottom " + Twine(CurrentFnSym->getName()) +
                          ".function");
}

void XCoreAsmPrinter::EmitFunctionEntryLabel() {
  // Mark the start of the function
  OutStreamer.EmitRawText("\t.cc_top " + Twine(CurrentFnSym->getName()) +
                          ".function," + CurrentFnSym->getName());
  OutStreamer.EmitLabel(CurrentFnSym);
}

void XCoreAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                      raw_ostream &O) {
  printOperand(MI, opNum, O);
  
  if (MI->getOperand(opNum+1).isImm() && MI->getOperand(opNum+1).getImm() == 0)
    return;
  
  O << "+";
  printOperand(MI, opNum+1, O);
}

void XCoreAsmPrinter::
printInlineJT(const MachineInstr *MI, int opNum, raw_ostream &O,
              const std::string &directive) {
  unsigned JTI = MI->getOperand(opNum).getIndex();
  const MachineFunction *MF = MI->getParent()->getParent();
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  O << "\t" << directive << " ";
  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    if (i > 0)
      O << ",";
    O << *MBB->getSymbol();
  }
}

void XCoreAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << getRegisterName(MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;
  case MachineOperand::MO_GlobalAddress:
    O << *Mang->getSymbol(MO.getGlobal());
    break;
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << MO.getIndex();
    break;
  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << MO.getIndex();
    break;
  case MachineOperand::MO_BlockAddress:
    O << *GetBlockAddressSymbol(MO.getBlockAddress());
    break;
  default:
    llvm_unreachable("not implemented");
  }
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool XCoreAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      unsigned AsmVariant,const char *ExtraCode,
                                      raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0])
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    }

printOperand(MI, OpNo, O);
  return false;
}

void XCoreAsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                             raw_ostream &OS) {
  unsigned NOps = MI->getNumOperands();
  assert(NOps == 4);
  OS << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps-1).getMetadata()));
  OS << V.getName();
  OS << " <- ";
  // Frame address.  Currently handles register +- offset only.
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
  OS << '['; printOperand(MI, 0, OS); OS << '+'; printOperand(MI, 1, OS);
  OS << ']';
  OS << "+";
  printOperand(MI, NOps-2, OS);
}

MachineLocation XCoreAsmPrinter::
getDebugValueLocation(const MachineInstr *MI) const {
  // Handles frame addresses emitted in XCoreInstrInfo::emitFrameIndexDebugValue.
  assert(MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm() &&
         "Unexpected MachineOperand types");
  return MachineLocation(MI->getOperand(0).getReg(),
                         MI->getOperand(1).getImm());
}

void XCoreAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> Str;
  raw_svector_ostream O(Str);

  switch (MI->getOpcode()) {
  case XCore::DBG_VALUE: {
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      SmallString<128> TmpStr;
      raw_svector_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
    return;
  }
  case XCore::ADD_2rus:
    if (MI->getOperand(2).getImm() == 0) {
      O << "\tmov " << getRegisterName(MI->getOperand(0).getReg()) << ", "
        << getRegisterName(MI->getOperand(1).getReg());
      OutStreamer.EmitRawText(O.str());
      return;
    }
    break;
  }
  printInstruction(MI, O);
  OutStreamer.EmitRawText(O.str());
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreAsmPrinter() { 
  RegisterAsmPrinter<XCoreAsmPrinter> X(TheXCoreTarget);
}
