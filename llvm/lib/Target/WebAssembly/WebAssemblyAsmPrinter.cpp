//===-- WebAssemblyAsmPrinter.cpp - WebAssembly LLVM assembly writer ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains a printer that converts from our internal
/// representation of machine-dependent LLVM code to the WebAssembly assembly
/// language.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMCInstLower.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class WebAssemblyAsmPrinter final : public AsmPrinter {
  const MachineRegisterInfo *MRI;
  const WebAssemblyFunctionInfo *MFI;

public:
  WebAssemblyAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), MRI(nullptr), MFI(nullptr) {}

private:
  const char *getPassName() const override {
    return "WebAssembly Assembly Printer";
  }

  //===------------------------------------------------------------------===//
  // MachineFunctionPass Implementation.
  //===------------------------------------------------------------------===//

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AsmPrinter::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    MRI = &MF.getRegInfo();
    MFI = MF.getInfo<WebAssemblyFunctionInfo>();
    return AsmPrinter::runOnMachineFunction(MF);
  }

  //===------------------------------------------------------------------===//
  // AsmPrinter Implementation.
  //===------------------------------------------------------------------===//

  void EmitJumpTableInfo() override;
  void EmitConstantPool() override;
  void EmitFunctionBodyStart() override;
  void EmitInstruction(const MachineInstr *MI) override;
  void EmitEndOfAsmFile(Module &M) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &OS) override;

  std::string getRegTypeName(unsigned RegNo) const;
  const char *toString(MVT VT) const;
  std::string regToString(const MachineOperand &MO);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

std::string WebAssemblyAsmPrinter::getRegTypeName(unsigned RegNo) const {
  const TargetRegisterClass *TRC = MRI->getRegClass(RegNo);
  for (MVT T : {MVT::i32, MVT::i64, MVT::f32, MVT::f64})
    if (TRC->hasType(T))
      return EVT(T).getEVTString();
  DEBUG(errs() << "Unknown type for register number: " << RegNo);
  llvm_unreachable("Unknown register type");
  return "?";
}

std::string WebAssemblyAsmPrinter::regToString(const MachineOperand &MO) {
  unsigned RegNo = MO.getReg();
  assert(TargetRegisterInfo::isVirtualRegister(RegNo) &&
         "Unlowered physical register encountered during assembly printing");
  assert(!MFI->isVRegStackified(RegNo));
  unsigned WAReg = MFI->getWAReg(RegNo);
  assert(WAReg != WebAssemblyFunctionInfo::UnusedReg);
  return '$' + utostr(WAReg);
}

const char *WebAssemblyAsmPrinter::toString(MVT VT) const {
  switch (VT.SimpleTy) {
  default:
    break;
  case MVT::f32:
    return "f32";
  case MVT::f64:
    return "f64";
  case MVT::i32:
    return "i32";
  case MVT::i64:
    return "i64";
  }
  DEBUG(dbgs() << "Invalid type " << EVT(VT).getEVTString() << '\n');
  llvm_unreachable("invalid type");
  return "<invalid>";
}

//===----------------------------------------------------------------------===//
// WebAssemblyAsmPrinter Implementation.
//===----------------------------------------------------------------------===//

void WebAssemblyAsmPrinter::EmitConstantPool() {
  assert(MF->getConstantPool()->getConstants().empty() &&
         "WebAssembly disables constant pools");
}

void WebAssemblyAsmPrinter::EmitJumpTableInfo() {
  // Nothing to do; jump tables are incorporated into the instruction stream.
}

static void ComputeLegalValueVTs(const Function &F,
                                 const TargetMachine &TM,
                                 Type *Ty,
                                 SmallVectorImpl<MVT> &ValueVTs) {
  const DataLayout& DL(F.getParent()->getDataLayout());
  const WebAssemblyTargetLowering &TLI =
      *TM.getSubtarget<WebAssemblySubtarget>(F).getTargetLowering();
  SmallVector<EVT, 4> VTs;
  ComputeValueVTs(TLI, DL, Ty, VTs);

  for (EVT VT : VTs) {
    unsigned NumRegs = TLI.getNumRegisters(F.getContext(), VT);
    MVT RegisterVT = TLI.getRegisterType(F.getContext(), VT);
    for (unsigned i = 0; i != NumRegs; ++i)
      ValueVTs.push_back(RegisterVT);
  }
}

void WebAssemblyAsmPrinter::EmitFunctionBodyStart() {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  for (MVT VT : MFI->getParams())
    OS << "\t" ".param " << toString(VT) << '\n';

  SmallVector<MVT, 4> ResultVTs;
  const Function &F(*MF->getFunction());
  ComputeLegalValueVTs(F, TM, F.getReturnType(), ResultVTs);
  // If the return type needs to be legalized it will get converted into
  // passing a pointer.
  if (ResultVTs.size() == 1)
    OS << "\t" ".result " << toString(ResultVTs.front()) << '\n';

  bool FirstWAReg = true;
  for (unsigned Idx = 0, IdxE = MRI->getNumVirtRegs(); Idx != IdxE; ++Idx) {
    unsigned VReg = TargetRegisterInfo::index2VirtReg(Idx);
    unsigned WAReg = MFI->getWAReg(VReg);
    // Don't declare unused registers.
    if (WAReg == WebAssemblyFunctionInfo::UnusedReg)
      continue;
    // Don't redeclare parameters.
    if (WAReg < MFI->getParams().size())
      continue;
    // Don't declare stackified registers.
    if (int(WAReg) < 0)
      continue;
    if (FirstWAReg)
      OS << "\t" ".local ";
    else
      OS << ", ";
    OS << getRegTypeName(VReg);
    FirstWAReg = false;
  }
  if (!FirstWAReg)
    OS << '\n';

  // EmitRawText appends a newline, so strip off the last newline.
  StringRef Text = OS.str();
  if (!Text.empty())
    OutStreamer->EmitRawText(Text.substr(0, Text.size() - 1));
  AsmPrinter::EmitFunctionBodyStart();
}

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  DEBUG(dbgs() << "EmitInstruction: " << *MI << '\n');

  switch (MI->getOpcode()) {
  case WebAssembly::ARGUMENT_I32:
  case WebAssembly::ARGUMENT_I64:
  case WebAssembly::ARGUMENT_F32:
  case WebAssembly::ARGUMENT_F64:
    // These represent values which are live into the function entry, so there's
    // no instruction to emit.
    break;
  default: {
    WebAssemblyMCInstLower MCInstLowering(OutContext, *this);
    MCInst TmpInst;
    MCInstLowering.Lower(MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);
    break;
  }
  }
}

void WebAssemblyAsmPrinter::EmitEndOfAsmFile(Module &M) {
  const DataLayout &DL = M.getDataLayout();

  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  for (const Function &F : M)
    if (F.isDeclarationForLinker()) {
      assert(F.hasName() && "imported functions must have a name");
      if (F.isIntrinsic())
        continue;
      if (Str.empty())
        OS << "\t.imports\n";

      MCSymbol *Sym = OutStreamer->getContext().getOrCreateSymbol(F.getName());
      OS << "\t.import " << *Sym << " \"\" " << *Sym;

      const WebAssemblyTargetLowering &TLI =
          *TM.getSubtarget<WebAssemblySubtarget>(F).getTargetLowering();

      // If we need to legalize the return type, it'll get converted into
      // passing a pointer.
      bool SawParam = false;
      SmallVector<MVT, 4> ResultVTs;
      ComputeLegalValueVTs(F, TM, F.getReturnType(), ResultVTs);
      if (ResultVTs.size() > 1) {
        ResultVTs.clear();
        OS << " (param " << toString(TLI.getPointerTy(DL));
        SawParam = true;
      }

      for (const Argument &A : F.args()) {
        SmallVector<MVT, 4> ParamVTs;
        ComputeLegalValueVTs(F, TM, A.getType(), ParamVTs);
        for (MVT VT : ParamVTs) {
          if (!SawParam) {
            OS << " (param";
            SawParam = true;
          }
          OS << ' ' << toString(VT);
        }
      }
      if (SawParam)
        OS << ')';

      for (MVT VT : ResultVTs)
        OS << " (result " << toString(VT) << ')';

      OS << '\n';
    }

  StringRef Text = OS.str();
  if (!Text.empty())
    OutStreamer->EmitRawText(Text.substr(0, Text.size() - 1));
}

bool WebAssemblyAsmPrinter::PrintAsmOperand(const MachineInstr *MI,
                                            unsigned OpNo, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  if (AsmVariant != 0)
    report_fatal_error("There are no defined alternate asm variants");

  if (!ExtraCode) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    if (MO.isImm())
      OS << MO.getImm();
    else
      OS << regToString(MO);
    return false;
  }

  return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, OS);
}

bool WebAssemblyAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                                  unsigned OpNo,
                                                  unsigned AsmVariant,
                                                  const char *ExtraCode,
                                                  raw_ostream &OS) {
  if (AsmVariant != 0)
    report_fatal_error("There are no defined alternate asm variants");

  if (!ExtraCode) {
    OS << regToString(MI->getOperand(OpNo));
    return false;
  }

  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, AsmVariant, ExtraCode, OS);
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(TheWebAssemblyTarget32);
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(TheWebAssemblyTarget64);
}
