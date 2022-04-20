//===-- SPIRVAsmPrinter.cpp - SPIR-V LLVM assembly writer ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the SPIR-V assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SPIRVInstPrinter.h"
#include "SPIRV.h"
#include "SPIRVInstrInfo.h"
#include "SPIRVMCInstLower.h"
#include "SPIRVModuleAnalysis.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "TargetInfo/SPIRVTargetInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class SPIRVAsmPrinter : public AsmPrinter {
public:
  explicit SPIRVAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), ST(nullptr), TII(nullptr) {}
  bool ModuleSectionsEmitted;
  const SPIRVSubtarget *ST;
  const SPIRVInstrInfo *TII;

  StringRef getPassName() const override { return "SPIRV Assembly Printer"; }
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;

  void outputMCInst(MCInst &Inst);
  void outputInstruction(const MachineInstr *MI);
  void outputModuleSection(SPIRV::ModuleSectionType MSType);
  void outputEntryPoints();
  void outputDebugSourceAndStrings(const Module &M);
  void outputOpMemoryModel();
  void outputOpFunctionEnd();
  void outputExtFuncDecls();
  void outputModuleSections();

  void emitInstruction(const MachineInstr *MI) override;
  void emitFunctionEntryLabel() override {}
  void emitFunctionHeader() override;
  void emitFunctionBodyStart() override {}
  void emitFunctionBodyEnd() override;
  void emitBasicBlockStart(const MachineBasicBlock &MBB) override;
  void emitBasicBlockEnd(const MachineBasicBlock &MBB) override {}
  void emitGlobalVariable(const GlobalVariable *GV) override {}
  void emitOpLabel(const MachineBasicBlock &MBB);
  void emitEndOfAsmFile(Module &M) override;
  bool doInitialization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  SPIRV::ModuleAnalysisInfo *MAI;
};
} // namespace

void SPIRVAsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<SPIRVModuleAnalysis>();
  AU.addPreserved<SPIRVModuleAnalysis>();
  AsmPrinter::getAnalysisUsage(AU);
}

// If the module has no functions, we need output global info anyway.
void SPIRVAsmPrinter::emitEndOfAsmFile(Module &M) {
  if (ModuleSectionsEmitted == false) {
    outputModuleSections();
    ModuleSectionsEmitted = true;
  }
}

void SPIRVAsmPrinter::emitFunctionHeader() {
  if (ModuleSectionsEmitted == false) {
    outputModuleSections();
    ModuleSectionsEmitted = true;
  }
  // Get the subtarget from the current MachineFunction.
  ST = &MF->getSubtarget<SPIRVSubtarget>();
  TII = ST->getInstrInfo();
  const Function &F = MF->getFunction();

  if (isVerbose()) {
    OutStreamer->GetCommentOS()
        << "-- Begin function "
        << GlobalValue::dropLLVMManglingEscape(F.getName()) << '\n';
  }

  auto Section = getObjFileLowering().SectionForGlobal(&F, TM);
  MF->setSection(Section);
}

void SPIRVAsmPrinter::outputOpFunctionEnd() {
  MCInst FunctionEndInst;
  FunctionEndInst.setOpcode(SPIRV::OpFunctionEnd);
  outputMCInst(FunctionEndInst);
}

// Emit OpFunctionEnd at the end of MF and clear BBNumToRegMap.
void SPIRVAsmPrinter::emitFunctionBodyEnd() {
  outputOpFunctionEnd();
  MAI->BBNumToRegMap.clear();
}

void SPIRVAsmPrinter::emitOpLabel(const MachineBasicBlock &MBB) {
  MCInst LabelInst;
  LabelInst.setOpcode(SPIRV::OpLabel);
  LabelInst.addOperand(MCOperand::createReg(MAI->getOrCreateMBBRegister(MBB)));
  outputMCInst(LabelInst);
}

void SPIRVAsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  // If it's the first MBB in MF, it has OpFunction and OpFunctionParameter, so
  // OpLabel should be output after them.
  if (MBB.getNumber() == MF->front().getNumber()) {
    for (const MachineInstr &MI : MBB)
      if (MI.getOpcode() == SPIRV::OpFunction)
        return;
    // TODO: this case should be checked by the verifier.
    report_fatal_error("OpFunction is expected in the front MBB of MF");
  }
  emitOpLabel(MBB);
}

void SPIRVAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << SPIRVInstPrinter::getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_FPImmediate:
    O << MO.getFPImm();
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

bool SPIRVAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Invalid instruction - SPIR-V does not have special modifiers

  printOperand(MI, OpNo, O);
  return false;
}

static bool isFuncOrHeaderInstr(const MachineInstr *MI,
                                const SPIRVInstrInfo *TII) {
  return TII->isHeaderInstr(*MI) || MI->getOpcode() == SPIRV::OpFunction ||
         MI->getOpcode() == SPIRV::OpFunctionParameter;
}

void SPIRVAsmPrinter::outputMCInst(MCInst &Inst) {
  OutStreamer->emitInstruction(Inst, *OutContext.getSubtargetInfo());
}

void SPIRVAsmPrinter::outputInstruction(const MachineInstr *MI) {
  SPIRVMCInstLower MCInstLowering;
  MCInst TmpInst;
  MCInstLowering.lower(MI, TmpInst, MAI);
  outputMCInst(TmpInst);
}

void SPIRVAsmPrinter::emitInstruction(const MachineInstr *MI) {
  if (!MAI->getSkipEmission(MI))
    outputInstruction(MI);

  // Output OpLabel after OpFunction and OpFunctionParameter in the first MBB.
  const MachineInstr *NextMI = MI->getNextNode();
  if (!MAI->hasMBBRegister(*MI->getParent()) && isFuncOrHeaderInstr(MI, TII) &&
      (!NextMI || !isFuncOrHeaderInstr(NextMI, TII))) {
    assert(MI->getParent()->getNumber() == MF->front().getNumber() &&
           "OpFunction is not in the front MBB of MF");
    emitOpLabel(*MI->getParent());
  }
}

void SPIRVAsmPrinter::outputModuleSection(SPIRV::ModuleSectionType MSType) {
  for (MachineInstr *MI : MAI->getMSInstrs(MSType))
    outputInstruction(MI);
}

void SPIRVAsmPrinter::outputDebugSourceAndStrings(const Module &M) {
  // Output OpSource.
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpSource);
  Inst.addOperand(MCOperand::createImm(static_cast<unsigned>(MAI->SrcLang)));
  Inst.addOperand(
      MCOperand::createImm(static_cast<unsigned>(MAI->SrcLangVersion)));
  outputMCInst(Inst);
}

void SPIRVAsmPrinter::outputOpMemoryModel() {
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpMemoryModel);
  Inst.addOperand(MCOperand::createImm(static_cast<unsigned>(MAI->Addr)));
  Inst.addOperand(MCOperand::createImm(static_cast<unsigned>(MAI->Mem)));
  outputMCInst(Inst);
}

// Before the OpEntryPoints' output, we need to add the entry point's
// interfaces. The interface is a list of IDs of global OpVariable instructions.
// These declare the set of global variables from a module that form
// the interface of this entry point.
void SPIRVAsmPrinter::outputEntryPoints() {
  // Find all OpVariable IDs with required StorageClass.
  DenseSet<Register> InterfaceIDs;
  for (MachineInstr *MI : MAI->GlobalVarList) {
    assert(MI->getOpcode() == SPIRV::OpVariable);
    auto SC = static_cast<SPIRV::StorageClass>(MI->getOperand(2).getImm());
    // Before version 1.4, the interface's storage classes are limited to
    // the Input and Output storage classes. Starting with version 1.4,
    // the interface's storage classes are all storage classes used in
    // declaring all global variables referenced by the entry point call tree.
    if (ST->getSPIRVVersion() >= 14 || SC == SPIRV::StorageClass::Input ||
        SC == SPIRV::StorageClass::Output) {
      MachineFunction *MF = MI->getMF();
      Register Reg = MAI->getRegisterAlias(MF, MI->getOperand(0).getReg());
      InterfaceIDs.insert(Reg);
    }
  }

  // Output OpEntryPoints adding interface args to all of them.
  for (MachineInstr *MI : MAI->getMSInstrs(SPIRV::MB_EntryPoints)) {
    SPIRVMCInstLower MCInstLowering;
    MCInst TmpInst;
    MCInstLowering.lower(MI, TmpInst, MAI);
    for (Register Reg : InterfaceIDs) {
      assert(Reg.isValid());
      TmpInst.addOperand(MCOperand::createReg(Reg));
    }
    outputMCInst(TmpInst);
  }
}

void SPIRVAsmPrinter::outputExtFuncDecls() {
  // Insert OpFunctionEnd after each declaration.
  SmallVectorImpl<MachineInstr *>::iterator
      I = MAI->getMSInstrs(SPIRV::MB_ExtFuncDecls).begin(),
      E = MAI->getMSInstrs(SPIRV::MB_ExtFuncDecls).end();
  for (; I != E; ++I) {
    outputInstruction(*I);
    if ((I + 1) == E || (*(I + 1))->getOpcode() == SPIRV::OpFunction)
      outputOpFunctionEnd();
  }
}

void SPIRVAsmPrinter::outputModuleSections() {
  const Module *M = MMI->getModule();
  // Get the global subtarget to output module-level info.
  ST = static_cast<const SPIRVTargetMachine &>(TM).getSubtargetImpl();
  TII = ST->getInstrInfo();
  MAI = &SPIRVModuleAnalysis::MAI;
  assert(ST && TII && MAI && M && "Module analysis is required");
  // Output instructions according to the Logical Layout of a Module:
  // TODO: 1,2. All OpCapability instructions, then optional OpExtension
  // instructions.
  // TODO: 3. Optional OpExtInstImport instructions.
  // 4. The single required OpMemoryModel instruction.
  outputOpMemoryModel();
  // 5. All entry point declarations, using OpEntryPoint.
  outputEntryPoints();
  // 6. Execution-mode declarations, using OpExecutionMode or OpExecutionModeId.
  // TODO:
  // 7a. Debug: all OpString, OpSourceExtension, OpSource, and
  // OpSourceContinued, without forward references.
  outputDebugSourceAndStrings(*M);
  // 7b. Debug: all OpName and all OpMemberName.
  outputModuleSection(SPIRV::MB_DebugNames);
  // 7c. Debug: all OpModuleProcessed instructions.
  outputModuleSection(SPIRV::MB_DebugModuleProcessed);
  // 8. All annotation instructions (all decorations).
  outputModuleSection(SPIRV::MB_Annotations);
  // 9. All type declarations (OpTypeXXX instructions), all constant
  // instructions, and all global variable declarations. This section is
  // the first section to allow use of: OpLine and OpNoLine debug information;
  // non-semantic instructions with OpExtInst.
  outputModuleSection(SPIRV::MB_TypeConstVars);
  // 10. All function declarations (functions without a body).
  outputExtFuncDecls();
  // 11. All function definitions (functions with a body).
  // This is done in regular function output.
}

bool SPIRVAsmPrinter::doInitialization(Module &M) {
  ModuleSectionsEmitted = false;
  // We need to call the parent's one explicitly.
  return AsmPrinter::doInitialization(M);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSPIRVAsmPrinter() {
  RegisterAsmPrinter<SPIRVAsmPrinter> X(getTheSPIRV32Target());
  RegisterAsmPrinter<SPIRVAsmPrinter> Y(getTheSPIRV64Target());
}
