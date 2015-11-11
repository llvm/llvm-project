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
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class WebAssemblyAsmPrinter final : public AsmPrinter {
  const WebAssemblyInstrInfo *TII;
  const MachineRegisterInfo *MRI;
  unsigned NumArgs;

public:
  WebAssemblyAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), TII(nullptr), MRI(nullptr) {}

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
    const auto &Subtarget = MF.getSubtarget<WebAssemblySubtarget>();
    TII = Subtarget.getInstrInfo();
    MRI = &MF.getRegInfo();
    NumArgs = MF.getInfo<WebAssemblyFunctionInfo>()->getParams().size();
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

  std::string getRegTypeName(unsigned RegNo) const;
  static std::string toString(const APFloat &APF);
  const char *toString(MVT VT) const;
  std::string regToString(const MachineOperand &MO);
  std::string argToString(const MachineOperand &MO);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

// Operand type (if any), followed by the lower-case version of the opcode's
// name matching the names WebAssembly opcodes are expected to have. The
// tablegen names are uppercase and suffixed with their type (after an
// underscore). Conversions are additionally prefixed with their input type
// (before a double underscore).
static std::string OpcodeName(const WebAssemblyInstrInfo *TII,
                              const MachineInstr *MI) {
  std::string N(StringRef(TII->getName(MI->getOpcode())).lower());
  std::string::size_type Len = N.length();
  std::string::size_type Under = N.rfind('_');
  bool HasType = std::string::npos != Under;
  std::string::size_type NameEnd = HasType ? Under : Len;
  std::string Name(&N[0], &N[NameEnd]);
  if (!HasType)
    return Name;
  for (const char *typelessOpcode : { "return", "call", "br_if" })
    if (Name == typelessOpcode)
      return Name;
  std::string Type(&N[NameEnd + 1], &N[Len]);
  std::string::size_type DoubleUnder = Name.find("__");
  bool IsConv = std::string::npos != DoubleUnder;
  if (!IsConv)
    return Type + '.' + Name;
  std::string InType(&Name[0], &Name[DoubleUnder]);
  return Type + '.' + std::string(&Name[DoubleUnder + 2], &Name[NameEnd]) +
      '/' + InType;
}

static std::string toSymbol(StringRef S) { return ("$" + S).str(); }

std::string WebAssemblyAsmPrinter::getRegTypeName(unsigned RegNo) const {
  const TargetRegisterClass *TRC = MRI->getRegClass(RegNo);
  for (MVT T : {MVT::i32, MVT::i64, MVT::f32, MVT::f64})
    if (TRC->hasType(T))
      return EVT(T).getEVTString();
  DEBUG(errs() << "Unknown type for register number: " << RegNo);
  llvm_unreachable("Unknown register type");
  return "?";
}

std::string WebAssemblyAsmPrinter::toString(const APFloat &FP) {
  static const size_t BufBytes = 128;
  char buf[BufBytes];
  if (FP.isNaN())
    assert((FP.bitwiseIsEqual(APFloat::getQNaN(FP.getSemantics())) ||
            FP.bitwiseIsEqual(
                APFloat::getQNaN(FP.getSemantics(), /*Negative=*/true))) &&
           "convertToHexString handles neither SNaN nor NaN payloads");
  // Use C99's hexadecimal floating-point representation.
  auto Written = FP.convertToHexString(
      buf, /*hexDigits=*/0, /*upperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return buf;
}

std::string WebAssemblyAsmPrinter::regToString(const MachineOperand &MO) {
  unsigned RegNo = MO.getReg();
  if (TargetRegisterInfo::isPhysicalRegister(RegNo))
    return WebAssemblyInstPrinter::getRegisterName(RegNo);

  // WebAssembly arguments and local variables are in the same index space, and
  // there are no explicit varargs, so we just add the number of arguments to
  // the virtual register number to get the local variable number.
  return utostr(TargetRegisterInfo::virtReg2Index(RegNo) + NumArgs);
}

std::string WebAssemblyAsmPrinter::argToString(const MachineOperand &MO) {
  unsigned ArgNo = MO.getImm();
  // Same as above, but we don't need to add NumArgs here.
  return utostr(ArgNo);
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

void WebAssemblyAsmPrinter::EmitFunctionBodyStart() {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  for (MVT VT : MF->getInfo<WebAssemblyFunctionInfo>()->getParams())
    OS << "\t" ".param "
       << toString(VT) << '\n';
  for (MVT VT : MF->getInfo<WebAssemblyFunctionInfo>()->getResults())
    OS << "\t" ".result "
       << toString(VT) << '\n';

  bool FirstVReg = true;
  for (unsigned Idx = 0, IdxE = MRI->getNumVirtRegs(); Idx != IdxE; ++Idx) {
    unsigned VReg = TargetRegisterInfo::index2VirtReg(Idx);
    // FIXME: Don't skip dead virtual registers for now: that would require
    //        remapping all locals' numbers.
    // if (!MRI->use_empty(VReg)) {
    if (FirstVReg)
      OS << "\t" ".local ";
    else
      OS << ", ";
    OS << getRegTypeName(VReg);
    FirstVReg = false;
    //}
  }
  if (!FirstVReg)
    OS << '\n';

  // EmitRawText appends a newline, so strip off the last newline.
  StringRef Text = OS.str();
  if (!Text.empty())
    OutStreamer->EmitRawText(Text.substr(0, Text.size() - 1));
  AsmPrinter::EmitFunctionBodyStart();
}

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  DEBUG(dbgs() << "EmitInstruction: " << *MI << '\n');
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  unsigned NumDefs = MI->getDesc().getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  OS << '\t';

  switch (MI->getOpcode()) {
  case TargetOpcode::COPY:
    OS << "get_local push, " << regToString(MI->getOperand(1));
    break;
  case WebAssembly::ARGUMENT_I32:
  case WebAssembly::ARGUMENT_I64:
  case WebAssembly::ARGUMENT_F32:
  case WebAssembly::ARGUMENT_F64:
    OS << "get_local push, " << argToString(MI->getOperand(1));
    break;
  default: {
    OS << OpcodeName(TII, MI);
    bool NeedComma = false;
    bool DefsPushed = false;
    if (NumDefs != 0 && !MI->isCall()) {
      OS << " push";
      NeedComma = true;
      DefsPushed = true;
    }
    for (const MachineOperand &MO : MI->uses()) {
      if (MO.isReg() && MO.isImplicit())
        continue;
      if (NeedComma)
        OS << ',';
      NeedComma = true;
      OS << ' ';
      switch (MO.getType()) {
      default:
        llvm_unreachable("unexpected machine operand type");
      case MachineOperand::MO_Register:
        OS << "(get_local " << regToString(MO) << ')';
        break;
      case MachineOperand::MO_Immediate:
        OS << MO.getImm();
        break;
      case MachineOperand::MO_FPImmediate:
        OS << toString(MO.getFPImm()->getValueAPF());
        break;
      case MachineOperand::MO_GlobalAddress:
        OS << toSymbol(MO.getGlobal()->getName());
        break;
      case MachineOperand::MO_MachineBasicBlock:
        OS << toSymbol(MO.getMBB()->getSymbol()->getName());
        break;
      }
      if (NumDefs != 0 && !DefsPushed) {
        // Special-case for calls; print the push after the callee.
        assert(MI->isCall());
        OS << ", push";
        DefsPushed = true;
      }
    }
    break;
  }
  }

  OutStreamer->EmitRawText(OS.str());

  if (NumDefs != 0) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    const MachineOperand &Operand = MI->getOperand(0);
    OS << "\tset_local " << regToString(Operand) << ", pop";
    OutStreamer->EmitRawText(OS.str());
  }
}

static void ComputeLegalValueVTs(LLVMContext &Context,
                                 const WebAssemblyTargetLowering &TLI,
                                 const DataLayout &DL, Type *Ty,
                                 SmallVectorImpl<MVT> &ValueVTs) {
  SmallVector<EVT, 4> VTs;
  ComputeValueVTs(TLI, DL, Ty, VTs);

  for (EVT VT : VTs) {
    unsigned NumRegs = TLI.getNumRegisters(Context, VT);
    MVT RegisterVT = TLI.getRegisterType(Context, VT);
    for (unsigned i = 0; i != NumRegs; ++i)
      ValueVTs.push_back(RegisterVT);
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

      OS << "\t.import " << toSymbol(F.getName()) << " \"\" \"" << F.getName()
         << "\"";

      const WebAssemblyTargetLowering &TLI =
          *TM.getSubtarget<WebAssemblySubtarget>(F).getTargetLowering();

      // If we need to legalize the return type, it'll get converted into
      // passing a pointer.
      bool SawParam = false;
      SmallVector<MVT, 4> ResultVTs;
      ComputeLegalValueVTs(M.getContext(), TLI, DL, F.getReturnType(),
                           ResultVTs);
      if (ResultVTs.size() > 1) {
        ResultVTs.clear();
        OS << " (param " << toString(TLI.getPointerTy(DL));
        SawParam = true;
      }

      for (const Argument &A : F.args()) {
        SmallVector<MVT, 4> ParamVTs;
        ComputeLegalValueVTs(M.getContext(), TLI, DL, A.getType(), ParamVTs);
        for (EVT VT : ParamVTs) {
          if (!SawParam) {
            OS << " (param";
            SawParam = true;
          }
          OS << ' ' << toString(VT.getSimpleVT());
        }
      }
      if (SawParam)
        OS << ')';

      for (EVT VT : ResultVTs)
        OS << " (result " << toString(VT.getSimpleVT()) << ')';

      OS << '\n';
    }

  StringRef Text = OS.str();
  if (!Text.empty())
    OutStreamer->EmitRawText(Text.substr(0, Text.size() - 1));
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(TheWebAssemblyTarget32);
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(TheWebAssemblyTarget64);
}
