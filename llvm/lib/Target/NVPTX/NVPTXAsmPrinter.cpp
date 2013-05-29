//===-- NVPTXAsmPrinter.cpp - NVPTX LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to NVPTX assembly language.
//
//===----------------------------------------------------------------------===//

#include "NVPTXAsmPrinter.h"
#include "MCTargetDesc/NVPTXMCAsmInfo.h"
#include "NVPTX.h"
#include "NVPTXInstrInfo.h"
#include "NVPTXNumRegisters.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "cl_common_defines.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TimeValue.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <sstream>
using namespace llvm;

#include "NVPTXGenAsmWriter.inc"

bool RegAllocNilUsed = true;

#define DEPOTNAME "__local_depot"

static cl::opt<bool>
EmitLineNumbers("nvptx-emit-line-numbers",
                cl::desc("NVPTX Specific: Emit Line numbers even without -G"),
                cl::init(true));

namespace llvm { bool InterleaveSrcInPtx = false; }

static cl::opt<bool, true>
InterleaveSrc("nvptx-emit-src", cl::ZeroOrMore,
              cl::desc("NVPTX Specific: Emit source line in ptx file"),
              cl::location(llvm::InterleaveSrcInPtx));

namespace {
/// DiscoverDependentGlobals - Return a set of GlobalVariables on which \p V
/// depends.
void DiscoverDependentGlobals(const Value *V,
                              DenseSet<const GlobalVariable *> &Globals) {
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    Globals.insert(GV);
  else {
    if (const User *U = dyn_cast<User>(V)) {
      for (unsigned i = 0, e = U->getNumOperands(); i != e; ++i) {
        DiscoverDependentGlobals(U->getOperand(i), Globals);
      }
    }
  }
}

/// VisitGlobalVariableForEmission - Add \p GV to the list of GlobalVariable
/// instances to be emitted, but only after any dependents have been added
/// first.
void VisitGlobalVariableForEmission(
    const GlobalVariable *GV, SmallVectorImpl<const GlobalVariable *> &Order,
    DenseSet<const GlobalVariable *> &Visited,
    DenseSet<const GlobalVariable *> &Visiting) {
  // Have we already visited this one?
  if (Visited.count(GV))
    return;

  // Do we have a circular dependency?
  if (Visiting.count(GV))
    report_fatal_error("Circular dependency found in global variable set");

  // Start visiting this global
  Visiting.insert(GV);

  // Make sure we visit all dependents first
  DenseSet<const GlobalVariable *> Others;
  for (unsigned i = 0, e = GV->getNumOperands(); i != e; ++i)
    DiscoverDependentGlobals(GV->getOperand(i), Others);

  for (DenseSet<const GlobalVariable *>::iterator I = Others.begin(),
                                                  E = Others.end();
       I != E; ++I)
    VisitGlobalVariableForEmission(*I, Order, Visited, Visiting);

  // Now we can visit ourself
  Order.push_back(GV);
  Visited.insert(GV);
  Visiting.erase(GV);
}
}

// @TODO: This is a copy from AsmPrinter.cpp.  The function is static, so we
// cannot just link to the existing version.
/// LowerConstant - Lower the specified LLVM Constant to an MCExpr.
///
using namespace nvptx;
const MCExpr *nvptx::LowerConstant(const Constant *CV, AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;

  if (CV->isNullValue() || isa<UndefValue>(CV))
    return MCConstantExpr::Create(0, Ctx);

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV))
    return MCConstantExpr::Create(CI->getZExtValue(), Ctx);

  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    return MCSymbolRefExpr::Create(AP.Mang->getSymbol(GV), Ctx);

  if (const BlockAddress *BA = dyn_cast<BlockAddress>(CV))
    return MCSymbolRefExpr::Create(AP.GetBlockAddressSymbol(BA), Ctx);

  const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV);
  if (CE == 0)
    llvm_unreachable("Unknown constant value to lower!");

  switch (CE->getOpcode()) {
  default:
    // If the code isn't optimized, there may be outstanding folding
    // opportunities. Attempt to fold the expression using DataLayout as a
    // last resort before giving up.
    if (Constant *C = ConstantFoldConstantExpression(CE, AP.TM.getDataLayout()))
      if (C != CE)
        return LowerConstant(C, AP);

    // Otherwise report the problem to the user.
    {
      std::string S;
      raw_string_ostream OS(S);
      OS << "Unsupported expression in static initializer: ";
      WriteAsOperand(OS, CE, /*PrintType=*/ false,
                     !AP.MF ? 0 : AP.MF->getFunction()->getParent());
      report_fatal_error(OS.str());
    }
  case Instruction::GetElementPtr: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Generate a symbolic expression for the byte address
    APInt OffsetAI(TD.getPointerSizeInBits(), 0);
    cast<GEPOperator>(CE)->accumulateConstantOffset(TD, OffsetAI);

    const MCExpr *Base = LowerConstant(CE->getOperand(0), AP);
    if (!OffsetAI)
      return Base;

    int64_t Offset = OffsetAI.getSExtValue();
    return MCBinaryExpr::CreateAdd(Base, MCConstantExpr::Create(Offset, Ctx),
                                   Ctx);
  }

  case Instruction::Trunc:
    // We emit the value and depend on the assembler to truncate the generated
    // expression properly.  This is important for differences between
    // blockaddress labels.  Since the two labels are in the same function, it
    // is reasonable to treat their delta as a 32-bit value.
  // FALL THROUGH.
  case Instruction::BitCast:
    return LowerConstant(CE->getOperand(0), AP);

  case Instruction::IntToPtr: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Handle casts to pointers by changing them into casts to the appropriate
    // integer type.  This promotes constant folding and simplifies this code.
    Constant *Op = CE->getOperand(0);
    Op = ConstantExpr::getIntegerCast(Op, TD.getIntPtrType(CV->getContext()),
                                      false /*ZExt*/);
    return LowerConstant(Op, AP);
  }

  case Instruction::PtrToInt: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Support only foldable casts to/from pointers that can be eliminated by
    // changing the pointer to the appropriately sized integer type.
    Constant *Op = CE->getOperand(0);
    Type *Ty = CE->getType();

    const MCExpr *OpExpr = LowerConstant(Op, AP);

    // We can emit the pointer value into this slot if the slot is an
    // integer slot equal to the size of the pointer.
    if (TD.getTypeAllocSize(Ty) == TD.getTypeAllocSize(Op->getType()))
      return OpExpr;

    // Otherwise the pointer is smaller than the resultant integer, mask off
    // the high bits so we are sure to get a proper truncation if the input is
    // a constant expr.
    unsigned InBits = TD.getTypeAllocSizeInBits(Op->getType());
    const MCExpr *MaskExpr =
        MCConstantExpr::Create(~0ULL >> (64 - InBits), Ctx);
    return MCBinaryExpr::CreateAnd(OpExpr, MaskExpr, Ctx);
  }

    // The MC library also has a right-shift operator, but it isn't consistently
  // signed or unsigned between different targets.
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::Shl:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    const MCExpr *LHS = LowerConstant(CE->getOperand(0), AP);
    const MCExpr *RHS = LowerConstant(CE->getOperand(1), AP);
    switch (CE->getOpcode()) {
    default:
      llvm_unreachable("Unknown binary operator constant cast expr");
    case Instruction::Add:
      return MCBinaryExpr::CreateAdd(LHS, RHS, Ctx);
    case Instruction::Sub:
      return MCBinaryExpr::CreateSub(LHS, RHS, Ctx);
    case Instruction::Mul:
      return MCBinaryExpr::CreateMul(LHS, RHS, Ctx);
    case Instruction::SDiv:
      return MCBinaryExpr::CreateDiv(LHS, RHS, Ctx);
    case Instruction::SRem:
      return MCBinaryExpr::CreateMod(LHS, RHS, Ctx);
    case Instruction::Shl:
      return MCBinaryExpr::CreateShl(LHS, RHS, Ctx);
    case Instruction::And:
      return MCBinaryExpr::CreateAnd(LHS, RHS, Ctx);
    case Instruction::Or:
      return MCBinaryExpr::CreateOr(LHS, RHS, Ctx);
    case Instruction::Xor:
      return MCBinaryExpr::CreateXor(LHS, RHS, Ctx);
    }
  }
  }
}

void NVPTXAsmPrinter::emitLineNumberAsDotLoc(const MachineInstr &MI) {
  if (!EmitLineNumbers)
    return;
  if (ignoreLoc(MI))
    return;

  DebugLoc curLoc = MI.getDebugLoc();

  if (prevDebugLoc.isUnknown() && curLoc.isUnknown())
    return;

  if (prevDebugLoc == curLoc)
    return;

  prevDebugLoc = curLoc;

  if (curLoc.isUnknown())
    return;

  const MachineFunction *MF = MI.getParent()->getParent();
  //const TargetMachine &TM = MF->getTarget();

  const LLVMContext &ctx = MF->getFunction()->getContext();
  DIScope Scope(curLoc.getScope(ctx));

  if (!Scope.Verify())
    return;

  StringRef fileName(Scope.getFilename());
  StringRef dirName(Scope.getDirectory());
  SmallString<128> FullPathName = dirName;
  if (!dirName.empty() && !sys::path::is_absolute(fileName)) {
    sys::path::append(FullPathName, fileName);
    fileName = FullPathName.str();
  }

  if (filenameMap.find(fileName.str()) == filenameMap.end())
    return;

  // Emit the line from the source file.
  if (llvm::InterleaveSrcInPtx)
    this->emitSrcInText(fileName.str(), curLoc.getLine());

  std::stringstream temp;
  temp << "\t.loc " << filenameMap[fileName.str()] << " " << curLoc.getLine()
       << " " << curLoc.getCol();
  OutStreamer.EmitRawText(Twine(temp.str().c_str()));
}

void NVPTXAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  if (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA)
    emitLineNumberAsDotLoc(*MI);
  printInstruction(MI, OS);
  OutStreamer.EmitRawText(OS.str());
}

void NVPTXAsmPrinter::printReturnValStr(const Function *F, raw_ostream &O) {
  const DataLayout *TD = TM.getDataLayout();
  const TargetLowering *TLI = TM.getTargetLowering();

  Type *Ty = F->getReturnType();

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);

  if (Ty->getTypeID() == Type::VoidTyID)
    return;

  O << " (";

  if (isABI) {
    if (Ty->isPrimitiveType() || Ty->isIntegerTy()) {
      unsigned size = 0;
      if (const IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
        size = ITy->getBitWidth();
        if (size < 32)
          size = 32;
      } else {
        assert(Ty->isFloatingPointTy() && "Floating point type expected here");
        size = Ty->getPrimitiveSizeInBits();
      }

      O << ".param .b" << size << " func_retval0";
    } else if (isa<PointerType>(Ty)) {
      O << ".param .b" << TLI->getPointerTy().getSizeInBits()
        << " func_retval0";
    } else {
      if ((Ty->getTypeID() == Type::StructTyID) || isa<VectorType>(Ty)) {
        SmallVector<EVT, 16> vtparts;
        ComputeValueVTs(*TLI, Ty, vtparts);
        unsigned totalsz = 0;
        for (unsigned i = 0, e = vtparts.size(); i != e; ++i) {
          unsigned elems = 1;
          EVT elemtype = vtparts[i];
          if (vtparts[i].isVector()) {
            elems = vtparts[i].getVectorNumElements();
            elemtype = vtparts[i].getVectorElementType();
          }
          for (unsigned j = 0, je = elems; j != je; ++j) {
            unsigned sz = elemtype.getSizeInBits();
            if (elemtype.isInteger() && (sz < 8))
              sz = 8;
            totalsz += sz / 8;
          }
        }
        unsigned retAlignment = 0;
        if (!llvm::getAlign(*F, 0, retAlignment))
          retAlignment = TD->getABITypeAlignment(Ty);
        O << ".param .align " << retAlignment << " .b8 func_retval0[" << totalsz
          << "]";
      } else
        assert(false && "Unknown return type");
    }
  } else {
    SmallVector<EVT, 16> vtparts;
    ComputeValueVTs(*TLI, Ty, vtparts);
    unsigned idx = 0;
    for (unsigned i = 0, e = vtparts.size(); i != e; ++i) {
      unsigned elems = 1;
      EVT elemtype = vtparts[i];
      if (vtparts[i].isVector()) {
        elems = vtparts[i].getVectorNumElements();
        elemtype = vtparts[i].getVectorElementType();
      }

      for (unsigned j = 0, je = elems; j != je; ++j) {
        unsigned sz = elemtype.getSizeInBits();
        if (elemtype.isInteger() && (sz < 32))
          sz = 32;
        O << ".reg .b" << sz << " func_retval" << idx;
        if (j < je - 1)
          O << ", ";
        ++idx;
      }
      if (i < e - 1)
        O << ", ";
    }
  }
  O << ") ";
  return;
}

void NVPTXAsmPrinter::printReturnValStr(const MachineFunction &MF,
                                        raw_ostream &O) {
  const Function *F = MF.getFunction();
  printReturnValStr(F, O);
}

void NVPTXAsmPrinter::EmitFunctionEntryLabel() {
  SmallString<128> Str;
  raw_svector_ostream O(Str);

  if (!GlobalsEmitted) {
    emitGlobals(*MF->getFunction()->getParent());
    GlobalsEmitted = true;
  }
  
  // Set up
  MRI = &MF->getRegInfo();
  F = MF->getFunction();
  emitLinkageDirective(F, O);
  if (llvm::isKernelFunction(*F))
    O << ".entry ";
  else {
    O << ".func ";
    printReturnValStr(*MF, O);
  }

  O << *CurrentFnSym;

  emitFunctionParamList(*MF, O);

  if (llvm::isKernelFunction(*F))
    emitKernelFunctionDirectives(*F, O);

  OutStreamer.EmitRawText(O.str());

  prevDebugLoc = DebugLoc();
}

void NVPTXAsmPrinter::EmitFunctionBodyStart() {
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();
  unsigned numRegClasses = TRI.getNumRegClasses();
  VRidGlobal2LocalMap = new std::map<unsigned, unsigned>[numRegClasses + 1];
  OutStreamer.EmitRawText(StringRef("{\n"));
  setAndEmitFunctionVirtualRegisters(*MF);

  SmallString<128> Str;
  raw_svector_ostream O(Str);
  emitDemotedVars(MF->getFunction(), O);
  OutStreamer.EmitRawText(O.str());
}

void NVPTXAsmPrinter::EmitFunctionBodyEnd() {
  OutStreamer.EmitRawText(StringRef("}\n"));
  delete[] VRidGlobal2LocalMap;
}

void NVPTXAsmPrinter::emitKernelFunctionDirectives(const Function &F,
                                                   raw_ostream &O) const {
  // If the NVVM IR has some of reqntid* specified, then output
  // the reqntid directive, and set the unspecified ones to 1.
  // If none of reqntid* is specified, don't output reqntid directive.
  unsigned reqntidx, reqntidy, reqntidz;
  bool specified = false;
  if (llvm::getReqNTIDx(F, reqntidx) == false)
    reqntidx = 1;
  else
    specified = true;
  if (llvm::getReqNTIDy(F, reqntidy) == false)
    reqntidy = 1;
  else
    specified = true;
  if (llvm::getReqNTIDz(F, reqntidz) == false)
    reqntidz = 1;
  else
    specified = true;

  if (specified)
    O << ".reqntid " << reqntidx << ", " << reqntidy << ", " << reqntidz
      << "\n";

  // If the NVVM IR has some of maxntid* specified, then output
  // the maxntid directive, and set the unspecified ones to 1.
  // If none of maxntid* is specified, don't output maxntid directive.
  unsigned maxntidx, maxntidy, maxntidz;
  specified = false;
  if (llvm::getMaxNTIDx(F, maxntidx) == false)
    maxntidx = 1;
  else
    specified = true;
  if (llvm::getMaxNTIDy(F, maxntidy) == false)
    maxntidy = 1;
  else
    specified = true;
  if (llvm::getMaxNTIDz(F, maxntidz) == false)
    maxntidz = 1;
  else
    specified = true;

  if (specified)
    O << ".maxntid " << maxntidx << ", " << maxntidy << ", " << maxntidz
      << "\n";

  unsigned mincta;
  if (llvm::getMinCTASm(F, mincta))
    O << ".minnctapersm " << mincta << "\n";
}

void NVPTXAsmPrinter::getVirtualRegisterName(unsigned vr, bool isVec,
                                             raw_ostream &O) {
  const TargetRegisterClass *RC = MRI->getRegClass(vr);
  unsigned id = RC->getID();

  std::map<unsigned, unsigned> &regmap = VRidGlobal2LocalMap[id];
  unsigned mapped_vr = regmap[vr];

  if (!isVec) {
    O << getNVPTXRegClassStr(RC) << mapped_vr;
    return;
  }
  report_fatal_error("Bad register!");
}

void NVPTXAsmPrinter::emitVirtualRegister(unsigned int vr, bool isVec,
                                          raw_ostream &O) {
  getVirtualRegisterName(vr, isVec, O);
}

void NVPTXAsmPrinter::printVecModifiedImmediate(
    const MachineOperand &MO, const char *Modifier, raw_ostream &O) {
  static const char vecelem[] = { '0', '1', '2', '3', '0', '1', '2', '3' };
  int Imm = (int) MO.getImm();
  if (0 == strcmp(Modifier, "vecelem"))
    O << "_" << vecelem[Imm];
  else if (0 == strcmp(Modifier, "vecv4comm1")) {
    if ((Imm < 0) || (Imm > 3))
      O << "//";
  } else if (0 == strcmp(Modifier, "vecv4comm2")) {
    if ((Imm < 4) || (Imm > 7))
      O << "//";
  } else if (0 == strcmp(Modifier, "vecv4pos")) {
    if (Imm < 0)
      Imm = 0;
    O << "_" << vecelem[Imm % 4];
  } else if (0 == strcmp(Modifier, "vecv2comm1")) {
    if ((Imm < 0) || (Imm > 1))
      O << "//";
  } else if (0 == strcmp(Modifier, "vecv2comm2")) {
    if ((Imm < 2) || (Imm > 3))
      O << "//";
  } else if (0 == strcmp(Modifier, "vecv2pos")) {
    if (Imm < 0)
      Imm = 0;
    O << "_" << vecelem[Imm % 2];
  } else
    llvm_unreachable("Unknown Modifier on immediate operand");
}

void NVPTXAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(opNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
      if (MO.getReg() == NVPTX::VRDepot)
        O << DEPOTNAME << getFunctionNumber();
      else
        O << getRegisterName(MO.getReg());
    } else {
      if (!Modifier)
        emitVirtualRegister(MO.getReg(), false, O);
      else {
        if (strcmp(Modifier, "vecfull") == 0)
          emitVirtualRegister(MO.getReg(), true, O);
        else
          llvm_unreachable(
              "Don't know how to handle the modifier on virtual register.");
      }
    }
    return;

  case MachineOperand::MO_Immediate:
    if (!Modifier)
      O << MO.getImm();
    else if (strstr(Modifier, "vec") == Modifier)
      printVecModifiedImmediate(MO, Modifier, O);
    else
      llvm_unreachable(
          "Don't know how to handle modifier on immediate operand");
    return;

  case MachineOperand::MO_FPImmediate:
    printFPConstant(MO.getFPImm(), O);
    break;

  case MachineOperand::MO_GlobalAddress:
    O << *Mang->getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_ExternalSymbol: {
    const char *symbname = MO.getSymbolName();
    if (strstr(symbname, ".PARAM") == symbname) {
      unsigned index;
      sscanf(symbname + 6, "%u[];", &index);
      printParamName(index, O);
    } else if (strstr(symbname, ".HLPPARAM") == symbname) {
      unsigned index;
      sscanf(symbname + 9, "%u[];", &index);
      O << *CurrentFnSym << "_param_" << index << "_offset";
    } else
      O << symbname;
    break;
  }

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;

  default:
    llvm_unreachable("Operand type not supported.");
  }
}

void NVPTXAsmPrinter::printImplicitDef(const MachineInstr *MI,
                                       raw_ostream &O) const {
#ifndef __OPTIMIZE__
  O << "\t// Implicit def :";
  //printOperand(MI, 0);
  O << "\n";
#endif
}

void NVPTXAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                      raw_ostream &O, const char *Modifier) {
  printOperand(MI, opNum, O);

  if (Modifier && !strcmp(Modifier, "add")) {
    O << ", ";
    printOperand(MI, opNum + 1, O);
  } else {
    if (MI->getOperand(opNum + 1).isImm() &&
        MI->getOperand(opNum + 1).getImm() == 0)
      return; // don't print ',0' or '+0'
    O << "+";
    printOperand(MI, opNum + 1, O);
  }
}

void NVPTXAsmPrinter::printLdStCode(const MachineInstr *MI, int opNum,
                                    raw_ostream &O, const char *Modifier) {
  if (Modifier) {
    const MachineOperand &MO = MI->getOperand(opNum);
    int Imm = (int) MO.getImm();
    if (!strcmp(Modifier, "volatile")) {
      if (Imm)
        O << ".volatile";
    } else if (!strcmp(Modifier, "addsp")) {
      switch (Imm) {
      case NVPTX::PTXLdStInstCode::GLOBAL:
        O << ".global";
        break;
      case NVPTX::PTXLdStInstCode::SHARED:
        O << ".shared";
        break;
      case NVPTX::PTXLdStInstCode::LOCAL:
        O << ".local";
        break;
      case NVPTX::PTXLdStInstCode::PARAM:
        O << ".param";
        break;
      case NVPTX::PTXLdStInstCode::CONSTANT:
        O << ".const";
        break;
      case NVPTX::PTXLdStInstCode::GENERIC:
        if (!nvptxSubtarget.hasGenericLdSt())
          O << ".global";
        break;
      default:
        llvm_unreachable("Wrong Address Space");
      }
    } else if (!strcmp(Modifier, "sign")) {
      if (Imm == NVPTX::PTXLdStInstCode::Signed)
        O << "s";
      else if (Imm == NVPTX::PTXLdStInstCode::Unsigned)
        O << "u";
      else
        O << "f";
    } else if (!strcmp(Modifier, "vec")) {
      if (Imm == NVPTX::PTXLdStInstCode::V2)
        O << ".v2";
      else if (Imm == NVPTX::PTXLdStInstCode::V4)
        O << ".v4";
    } else
      llvm_unreachable("Unknown Modifier");
  } else
    llvm_unreachable("Empty Modifier");
}

void NVPTXAsmPrinter::emitDeclaration(const Function *F, raw_ostream &O) {

  emitLinkageDirective(F, O);
  if (llvm::isKernelFunction(*F))
    O << ".entry ";
  else
    O << ".func ";
  printReturnValStr(F, O);
  O << *Mang->getSymbol(F) << "\n";
  emitFunctionParamList(F, O);
  O << ";\n";
}

static bool usedInGlobalVarDef(const Constant *C) {
  if (!C)
    return false;

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (GV->getName().str() == "llvm.used")
      return false;
    return true;
  }

  for (Value::const_use_iterator ui = C->use_begin(), ue = C->use_end();
       ui != ue; ++ui) {
    const Constant *C = dyn_cast<Constant>(*ui);
    if (usedInGlobalVarDef(C))
      return true;
  }
  return false;
}

static bool usedInOneFunc(const User *U, Function const *&oneFunc) {
  if (const GlobalVariable *othergv = dyn_cast<GlobalVariable>(U)) {
    if (othergv->getName().str() == "llvm.used")
      return true;
  }

  if (const Instruction *instr = dyn_cast<Instruction>(U)) {
    if (instr->getParent() && instr->getParent()->getParent()) {
      const Function *curFunc = instr->getParent()->getParent();
      if (oneFunc && (curFunc != oneFunc))
        return false;
      oneFunc = curFunc;
      return true;
    } else
      return false;
  }

  if (const MDNode *md = dyn_cast<MDNode>(U))
    if (md->hasName() && ((md->getName().str() == "llvm.dbg.gv") ||
                          (md->getName().str() == "llvm.dbg.sp")))
      return true;

  for (User::const_use_iterator ui = U->use_begin(), ue = U->use_end();
       ui != ue; ++ui) {
    if (usedInOneFunc(*ui, oneFunc) == false)
      return false;
  }
  return true;
}

/* Find out if a global variable can be demoted to local scope.
 * Currently, this is valid for CUDA shared variables, which have local
 * scope and global lifetime. So the conditions to check are :
 * 1. Is the global variable in shared address space?
 * 2. Does it have internal linkage?
 * 3. Is the global variable referenced only in one function?
 */
static bool canDemoteGlobalVar(const GlobalVariable *gv, Function const *&f) {
  if (gv->hasInternalLinkage() == false)
    return false;
  const PointerType *Pty = gv->getType();
  if (Pty->getAddressSpace() != llvm::ADDRESS_SPACE_SHARED)
    return false;

  const Function *oneFunc = 0;

  bool flag = usedInOneFunc(gv, oneFunc);
  if (flag == false)
    return false;
  if (!oneFunc)
    return false;
  f = oneFunc;
  return true;
}

static bool useFuncSeen(const Constant *C,
                        llvm::DenseMap<const Function *, bool> &seenMap) {
  for (Value::const_use_iterator ui = C->use_begin(), ue = C->use_end();
       ui != ue; ++ui) {
    if (const Constant *cu = dyn_cast<Constant>(*ui)) {
      if (useFuncSeen(cu, seenMap))
        return true;
    } else if (const Instruction *I = dyn_cast<Instruction>(*ui)) {
      const BasicBlock *bb = I->getParent();
      if (!bb)
        continue;
      const Function *caller = bb->getParent();
      if (!caller)
        continue;
      if (seenMap.find(caller) != seenMap.end())
        return true;
    }
  }
  return false;
}

void NVPTXAsmPrinter::emitDeclarations(const Module &M, raw_ostream &O) {
  llvm::DenseMap<const Function *, bool> seenMap;
  for (Module::const_iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    const Function *F = FI;

    if (F->isDeclaration()) {
      if (F->use_empty())
        continue;
      if (F->getIntrinsicID())
        continue;
      emitDeclaration(F, O);
      continue;
    }
    for (Value::const_use_iterator iter = F->use_begin(),
                                   iterEnd = F->use_end();
         iter != iterEnd; ++iter) {
      if (const Constant *C = dyn_cast<Constant>(*iter)) {
        if (usedInGlobalVarDef(C)) {
          // The use is in the initialization of a global variable
          // that is a function pointer, so print a declaration
          // for the original function
          emitDeclaration(F, O);
          break;
        }
        // Emit a declaration of this function if the function that
        // uses this constant expr has already been seen.
        if (useFuncSeen(C, seenMap)) {
          emitDeclaration(F, O);
          break;
        }
      }

      if (!isa<Instruction>(*iter))
        continue;
      const Instruction *instr = cast<Instruction>(*iter);
      const BasicBlock *bb = instr->getParent();
      if (!bb)
        continue;
      const Function *caller = bb->getParent();
      if (!caller)
        continue;

      // If a caller has already been seen, then the caller is
      // appearing in the module before the callee. so print out
      // a declaration for the callee.
      if (seenMap.find(caller) != seenMap.end()) {
        emitDeclaration(F, O);
        break;
      }
    }
    seenMap[F] = true;
  }
}

void NVPTXAsmPrinter::recordAndEmitFilenames(Module &M) {
  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(M);

  unsigned i = 1;
  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
                                 E = DbgFinder.compile_unit_end();
       I != E; ++I) {
    DICompileUnit DIUnit(*I);
    StringRef Filename(DIUnit.getFilename());
    StringRef Dirname(DIUnit.getDirectory());
    SmallString<128> FullPathName = Dirname;
    if (!Dirname.empty() && !sys::path::is_absolute(Filename)) {
      sys::path::append(FullPathName, Filename);
      Filename = FullPathName.str();
    }
    if (filenameMap.find(Filename.str()) != filenameMap.end())
      continue;
    filenameMap[Filename.str()] = i;
    OutStreamer.EmitDwarfFileDirective(i, "", Filename.str());
    ++i;
  }

  for (DebugInfoFinder::iterator I = DbgFinder.subprogram_begin(),
                                 E = DbgFinder.subprogram_end();
       I != E; ++I) {
    DISubprogram SP(*I);
    StringRef Filename(SP.getFilename());
    StringRef Dirname(SP.getDirectory());
    SmallString<128> FullPathName = Dirname;
    if (!Dirname.empty() && !sys::path::is_absolute(Filename)) {
      sys::path::append(FullPathName, Filename);
      Filename = FullPathName.str();
    }
    if (filenameMap.find(Filename.str()) != filenameMap.end())
      continue;
    filenameMap[Filename.str()] = i;
    ++i;
  }
}

bool NVPTXAsmPrinter::doInitialization(Module &M) {

  SmallString<128> Str1;
  raw_svector_ostream OS1(Str1);

  MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  MMI->AnalyzeModule(M);

  // We need to call the parent's one explicitly.
  //bool Result = AsmPrinter::doInitialization(M);

  // Initialize TargetLoweringObjectFile.
  const_cast<TargetLoweringObjectFile &>(getObjFileLowering())
      .Initialize(OutContext, TM);

  Mang = new Mangler(OutContext, *TM.getDataLayout());

  // Emit header before any dwarf directives are emitted below.
  emitHeader(M, OS1);
  OutStreamer.EmitRawText(OS1.str());

  // Already commented out
  //bool Result = AsmPrinter::doInitialization(M);

  if (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA)
    recordAndEmitFilenames(M);

  GlobalsEmitted = false;
    
  return false; // success
}

void NVPTXAsmPrinter::emitGlobals(const Module &M) {
  SmallString<128> Str2;
  raw_svector_ostream OS2(Str2);

  emitDeclarations(M, OS2);

  // As ptxas does not support forward references of globals, we need to first
  // sort the list of module-level globals in def-use order. We visit each
  // global variable in order, and ensure that we emit it *after* its dependent
  // globals. We use a little extra memory maintaining both a set and a list to
  // have fast searches while maintaining a strict ordering.
  SmallVector<const GlobalVariable *, 8> Globals;
  DenseSet<const GlobalVariable *> GVVisited;
  DenseSet<const GlobalVariable *> GVVisiting;

  // Visit each global variable, in order
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    VisitGlobalVariableForEmission(I, Globals, GVVisited, GVVisiting);

  assert(GVVisited.size() == M.getGlobalList().size() &&
         "Missed a global variable");
  assert(GVVisiting.size() == 0 && "Did not fully process a global variable");

  // Print out module-level global variables in proper order
  for (unsigned i = 0, e = Globals.size(); i != e; ++i)
    printModuleLevelGV(Globals[i], OS2);

  OS2 << '\n';

  OutStreamer.EmitRawText(OS2.str());
}

void NVPTXAsmPrinter::emitHeader(Module &M, raw_ostream &O) {
  O << "//\n";
  O << "// Generated by LLVM NVPTX Back-End\n";
  O << "//\n";
  O << "\n";

  unsigned PTXVersion = nvptxSubtarget.getPTXVersion();
  O << ".version " << (PTXVersion / 10) << "." << (PTXVersion % 10) << "\n";

  O << ".target ";
  O << nvptxSubtarget.getTargetName();

  if (nvptxSubtarget.getDrvInterface() == NVPTX::NVCL)
    O << ", texmode_independent";
  if (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA) {
    if (!nvptxSubtarget.hasDouble())
      O << ", map_f64_to_f32";
  }

  if (MAI->doesSupportDebugInformation())
    O << ", debug";

  O << "\n";

  O << ".address_size ";
  if (nvptxSubtarget.is64Bit())
    O << "64";
  else
    O << "32";
  O << "\n";

  O << "\n";
}

bool NVPTXAsmPrinter::doFinalization(Module &M) {

  // If we did not emit any functions, then the global declarations have not
  // yet been emitted.
  if (!GlobalsEmitted) {
    emitGlobals(M);
    GlobalsEmitted = true;
  }

  // XXX Temproarily remove global variables so that doFinalization() will not
  // emit them again (global variables are emitted at beginning).

  Module::GlobalListType &global_list = M.getGlobalList();
  int i, n = global_list.size();
  GlobalVariable **gv_array = new GlobalVariable *[n];

  // first, back-up GlobalVariable in gv_array
  i = 0;
  for (Module::global_iterator I = global_list.begin(), E = global_list.end();
       I != E; ++I)
    gv_array[i++] = &*I;

  // second, empty global_list
  while (!global_list.empty())
    global_list.remove(global_list.begin());

  // call doFinalization
  bool ret = AsmPrinter::doFinalization(M);

  // now we restore global variables
  for (i = 0; i < n; i++)
    global_list.insert(global_list.end(), gv_array[i]);

  delete[] gv_array;
  return ret;

  //bool Result = AsmPrinter::doFinalization(M);
  // Instead of calling the parents doFinalization, we may
  // clone parents doFinalization and customize here.
  // Currently, we if NVISA out the EmitGlobals() in
  // parent's doFinalization, which is too intrusive.
  //
  // Same for the doInitialization.
  //return Result;
}

// This function emits appropriate linkage directives for
// functions and global variables.
//
// extern function declaration            -> .extern
// extern function definition             -> .visible
// external global variable with init     -> .visible
// external without init                  -> .extern
// appending                              -> not allowed, assert.

void NVPTXAsmPrinter::emitLinkageDirective(const GlobalValue *V,
                                           raw_ostream &O) {
  if (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA) {
    if (V->hasExternalLinkage()) {
      if (isa<GlobalVariable>(V)) {
        const GlobalVariable *GVar = cast<GlobalVariable>(V);
        if (GVar) {
          if (GVar->hasInitializer())
            O << ".visible ";
          else
            O << ".extern ";
        }
      } else if (V->isDeclaration())
        O << ".extern ";
      else
        O << ".visible ";
    } else if (V->hasAppendingLinkage()) {
      std::string msg;
      msg.append("Error: ");
      msg.append("Symbol ");
      if (V->hasName())
        msg.append(V->getName().str());
      msg.append("has unsupported appending linkage type");
      llvm_unreachable(msg.c_str());
    }
  }
}

void NVPTXAsmPrinter::printModuleLevelGV(const GlobalVariable *GVar,
                                         raw_ostream &O,
                                         bool processDemoted) {

  // Skip meta data
  if (GVar->hasSection()) {
    if (GVar->getSection() == "llvm.metadata")
      return;
  }

  const DataLayout *TD = TM.getDataLayout();

  // GlobalVariables are always constant pointers themselves.
  const PointerType *PTy = GVar->getType();
  Type *ETy = PTy->getElementType();

  if (GVar->hasExternalLinkage()) {
    if (GVar->hasInitializer())
      O << ".visible ";
    else
      O << ".extern ";
  }

  if (llvm::isTexture(*GVar)) {
    O << ".global .texref " << llvm::getTextureName(*GVar) << ";\n";
    return;
  }

  if (llvm::isSurface(*GVar)) {
    O << ".global .surfref " << llvm::getSurfaceName(*GVar) << ";\n";
    return;
  }

  if (GVar->isDeclaration()) {
    // (extern) declarations, no definition or initializer
    // Currently the only known declaration is for an automatic __local
    // (.shared) promoted to global.
    emitPTXGlobalVariable(GVar, O);
    O << ";\n";
    return;
  }

  if (llvm::isSampler(*GVar)) {
    O << ".global .samplerref " << llvm::getSamplerName(*GVar);

    const Constant *Initializer = NULL;
    if (GVar->hasInitializer())
      Initializer = GVar->getInitializer();
    const ConstantInt *CI = NULL;
    if (Initializer)
      CI = dyn_cast<ConstantInt>(Initializer);
    if (CI) {
      unsigned sample = CI->getZExtValue();

      O << " = { ";

      for (int i = 0,
               addr = ((sample & __CLK_ADDRESS_MASK) >> __CLK_ADDRESS_BASE);
           i < 3; i++) {
        O << "addr_mode_" << i << " = ";
        switch (addr) {
        case 0:
          O << "wrap";
          break;
        case 1:
          O << "clamp_to_border";
          break;
        case 2:
          O << "clamp_to_edge";
          break;
        case 3:
          O << "wrap";
          break;
        case 4:
          O << "mirror";
          break;
        }
        O << ", ";
      }
      O << "filter_mode = ";
      switch ((sample & __CLK_FILTER_MASK) >> __CLK_FILTER_BASE) {
      case 0:
        O << "nearest";
        break;
      case 1:
        O << "linear";
        break;
      case 2:
        assert(0 && "Anisotropic filtering is not supported");
      default:
        O << "nearest";
        break;
      }
      if (!((sample & __CLK_NORMALIZED_MASK) >> __CLK_NORMALIZED_BASE)) {
        O << ", force_unnormalized_coords = 1";
      }
      O << " }";
    }

    O << ";\n";
    return;
  }

  if (GVar->hasPrivateLinkage()) {

    if (!strncmp(GVar->getName().data(), "unrollpragma", 12))
      return;

    // FIXME - need better way (e.g. Metadata) to avoid generating this global
    if (!strncmp(GVar->getName().data(), "filename", 8))
      return;
    if (GVar->use_empty())
      return;
  }

  const Function *demotedFunc = 0;
  if (!processDemoted && canDemoteGlobalVar(GVar, demotedFunc)) {
    O << "// " << GVar->getName().str() << " has been demoted\n";
    if (localDecls.find(demotedFunc) != localDecls.end())
      localDecls[demotedFunc].push_back(GVar);
    else {
      std::vector<const GlobalVariable *> temp;
      temp.push_back(GVar);
      localDecls[demotedFunc] = temp;
    }
    return;
  }

  O << ".";
  emitPTXAddressSpace(PTy->getAddressSpace(), O);
  if (GVar->getAlignment() == 0)
    O << " .align " << (int) TD->getPrefTypeAlignment(ETy);
  else
    O << " .align " << GVar->getAlignment();

  if (ETy->isPrimitiveType() || ETy->isIntegerTy() || isa<PointerType>(ETy)) {
    O << " .";
    // Special case: ABI requires that we use .u8 for predicates
    if (ETy->isIntegerTy(1))
      O << "u8";
    else
      O << getPTXFundamentalTypeStr(ETy, false);
    O << " ";
    O << *Mang->getSymbol(GVar);

    // Ptx allows variable initilization only for constant and global state
    // spaces.
    if (((PTy->getAddressSpace() == llvm::ADDRESS_SPACE_GLOBAL) ||
         (PTy->getAddressSpace() == llvm::ADDRESS_SPACE_CONST_NOT_GEN) ||
         (PTy->getAddressSpace() == llvm::ADDRESS_SPACE_CONST)) &&
        GVar->hasInitializer()) {
      const Constant *Initializer = GVar->getInitializer();
      if (!Initializer->isNullValue()) {
        O << " = ";
        printScalarConstant(Initializer, O);
      }
    }
  } else {
    unsigned int ElementSize = 0;

    // Although PTX has direct support for struct type and array type and
    // LLVM IR is very similar to PTX, the LLVM CodeGen does not support for
    // targets that support these high level field accesses. Structs, arrays
    // and vectors are lowered into arrays of bytes.
    switch (ETy->getTypeID()) {
    case Type::StructTyID:
    case Type::ArrayTyID:
    case Type::VectorTyID:
      ElementSize = TD->getTypeStoreSize(ETy);
      // Ptx allows variable initilization only for constant and
      // global state spaces.
      if (((PTy->getAddressSpace() == llvm::ADDRESS_SPACE_GLOBAL) ||
           (PTy->getAddressSpace() == llvm::ADDRESS_SPACE_CONST_NOT_GEN) ||
           (PTy->getAddressSpace() == llvm::ADDRESS_SPACE_CONST)) &&
          GVar->hasInitializer()) {
        const Constant *Initializer = GVar->getInitializer();
        if (!isa<UndefValue>(Initializer) && !Initializer->isNullValue()) {
          AggBuffer aggBuffer(ElementSize, O, *this);
          bufferAggregateConstant(Initializer, &aggBuffer);
          if (aggBuffer.numSymbols) {
            if (nvptxSubtarget.is64Bit()) {
              O << " .u64 " << *Mang->getSymbol(GVar) << "[";
              O << ElementSize / 8;
            } else {
              O << " .u32 " << *Mang->getSymbol(GVar) << "[";
              O << ElementSize / 4;
            }
            O << "]";
          } else {
            O << " .b8 " << *Mang->getSymbol(GVar) << "[";
            O << ElementSize;
            O << "]";
          }
          O << " = {";
          aggBuffer.print();
          O << "}";
        } else {
          O << " .b8 " << *Mang->getSymbol(GVar);
          if (ElementSize) {
            O << "[";
            O << ElementSize;
            O << "]";
          }
        }
      } else {
        O << " .b8 " << *Mang->getSymbol(GVar);
        if (ElementSize) {
          O << "[";
          O << ElementSize;
          O << "]";
        }
      }
      break;
    default:
      assert(0 && "type not supported yet");
    }

  }
  O << ";\n";
}

void NVPTXAsmPrinter::emitDemotedVars(const Function *f, raw_ostream &O) {
  if (localDecls.find(f) == localDecls.end())
    return;

  std::vector<const GlobalVariable *> &gvars = localDecls[f];

  for (unsigned i = 0, e = gvars.size(); i != e; ++i) {
    O << "\t// demoted variable\n\t";
    printModuleLevelGV(gvars[i], O, true);
  }
}

void NVPTXAsmPrinter::emitPTXAddressSpace(unsigned int AddressSpace,
                                          raw_ostream &O) const {
  switch (AddressSpace) {
  case llvm::ADDRESS_SPACE_LOCAL:
    O << "local";
    break;
  case llvm::ADDRESS_SPACE_GLOBAL:
    O << "global";
    break;
  case llvm::ADDRESS_SPACE_CONST:
    // This logic should be consistent with that in
    // getCodeAddrSpace() (NVPTXISelDATToDAT.cpp)
    if (nvptxSubtarget.hasGenericLdSt())
      O << "global";
    else
      O << "const";
    break;
  case llvm::ADDRESS_SPACE_CONST_NOT_GEN:
    O << "const";
    break;
  case llvm::ADDRESS_SPACE_SHARED:
    O << "shared";
    break;
  default:
    report_fatal_error("Bad address space found while emitting PTX");
    break;
  }
}

std::string
NVPTXAsmPrinter::getPTXFundamentalTypeStr(const Type *Ty, bool useB4PTR) const {
  switch (Ty->getTypeID()) {
  default:
    llvm_unreachable("unexpected type");
    break;
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1)
      return "pred";
    else if (NumBits <= 64) {
      std::string name = "u";
      return name + utostr(NumBits);
    } else {
      llvm_unreachable("Integer too large");
      break;
    }
    break;
  }
  case Type::FloatTyID:
    return "f32";
  case Type::DoubleTyID:
    return "f64";
  case Type::PointerTyID:
    if (nvptxSubtarget.is64Bit())
      if (useB4PTR)
        return "b64";
      else
        return "u64";
    else if (useB4PTR)
      return "b32";
    else
      return "u32";
  }
  llvm_unreachable("unexpected type");
  return NULL;
}

void NVPTXAsmPrinter::emitPTXGlobalVariable(const GlobalVariable *GVar,
                                            raw_ostream &O) {

  const DataLayout *TD = TM.getDataLayout();

  // GlobalVariables are always constant pointers themselves.
  const PointerType *PTy = GVar->getType();
  Type *ETy = PTy->getElementType();

  O << ".";
  emitPTXAddressSpace(PTy->getAddressSpace(), O);
  if (GVar->getAlignment() == 0)
    O << " .align " << (int) TD->getPrefTypeAlignment(ETy);
  else
    O << " .align " << GVar->getAlignment();

  if (ETy->isPrimitiveType() || ETy->isIntegerTy() || isa<PointerType>(ETy)) {
    O << " .";
    O << getPTXFundamentalTypeStr(ETy);
    O << " ";
    O << *Mang->getSymbol(GVar);
    return;
  }

  int64_t ElementSize = 0;

  // Although PTX has direct support for struct type and array type and LLVM IR
  // is very similar to PTX, the LLVM CodeGen does not support for targets that
  // support these high level field accesses. Structs and arrays are lowered
  // into arrays of bytes.
  switch (ETy->getTypeID()) {
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
    ElementSize = TD->getTypeStoreSize(ETy);
    O << " .b8 " << *Mang->getSymbol(GVar) << "[";
    if (ElementSize) {
      O << itostr(ElementSize);
    }
    O << "]";
    break;
  default:
    assert(0 && "type not supported yet");
  }
  return;
}

static unsigned int getOpenCLAlignment(const DataLayout *TD, Type *Ty) {
  if (Ty->isPrimitiveType() || Ty->isIntegerTy() || isa<PointerType>(Ty))
    return TD->getPrefTypeAlignment(Ty);

  const ArrayType *ATy = dyn_cast<ArrayType>(Ty);
  if (ATy)
    return getOpenCLAlignment(TD, ATy->getElementType());

  const VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (VTy) {
    Type *ETy = VTy->getElementType();
    unsigned int numE = VTy->getNumElements();
    unsigned int alignE = TD->getPrefTypeAlignment(ETy);
    if (numE == 3)
      return 4 * alignE;
    else
      return numE * alignE;
  }

  const StructType *STy = dyn_cast<StructType>(Ty);
  if (STy) {
    unsigned int alignStruct = 1;
    // Go through each element of the struct and find the
    // largest alignment.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; i++) {
      Type *ETy = STy->getElementType(i);
      unsigned int align = getOpenCLAlignment(TD, ETy);
      if (align > alignStruct)
        alignStruct = align;
    }
    return alignStruct;
  }

  const FunctionType *FTy = dyn_cast<FunctionType>(Ty);
  if (FTy)
    return TD->getPointerPrefAlignment();
  return TD->getPrefTypeAlignment(Ty);
}

void NVPTXAsmPrinter::printParamName(Function::const_arg_iterator I,
                                     int paramIndex, raw_ostream &O) {
  if ((nvptxSubtarget.getDrvInterface() == NVPTX::NVCL) ||
      (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA))
    O << *Mang->getSymbol(I->getParent()) << "_param_" << paramIndex;
  else {
    std::string argName = I->getName();
    const char *p = argName.c_str();
    while (*p) {
      if (*p == '.')
        O << "_";
      else
        O << *p;
      p++;
    }
  }
}

void NVPTXAsmPrinter::printParamName(int paramIndex, raw_ostream &O) {
  Function::const_arg_iterator I, E;
  int i = 0;

  if ((nvptxSubtarget.getDrvInterface() == NVPTX::NVCL) ||
      (nvptxSubtarget.getDrvInterface() == NVPTX::CUDA)) {
    O << *CurrentFnSym << "_param_" << paramIndex;
    return;
  }

  for (I = F->arg_begin(), E = F->arg_end(); I != E; ++I, i++) {
    if (i == paramIndex) {
      printParamName(I, paramIndex, O);
      return;
    }
  }
  llvm_unreachable("paramIndex out of bound");
}

void NVPTXAsmPrinter::emitFunctionParamList(const Function *F, raw_ostream &O) {
  const DataLayout *TD = TM.getDataLayout();
  const AttributeSet &PAL = F->getAttributes();
  const TargetLowering *TLI = TM.getTargetLowering();
  Function::const_arg_iterator I, E;
  unsigned paramIndex = 0;
  bool first = true;
  bool isKernelFunc = llvm::isKernelFunction(*F);
  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);
  MVT thePointerTy = TLI->getPointerTy();

  O << "(\n";

  for (I = F->arg_begin(), E = F->arg_end(); I != E; ++I, paramIndex++) {
    Type *Ty = I->getType();

    if (!first)
      O << ",\n";

    first = false;

    // Handle image/sampler parameters
    if (llvm::isSampler(*I) || llvm::isImage(*I)) {
      if (llvm::isImage(*I)) {
        std::string sname = I->getName();
        if (llvm::isImageWriteOnly(*I))
          O << "\t.param .surfref " << *Mang->getSymbol(F) << "_param_"
            << paramIndex;
        else // Default image is read_only
          O << "\t.param .texref " << *Mang->getSymbol(F) << "_param_"
            << paramIndex;
      } else // Should be llvm::isSampler(*I)
        O << "\t.param .samplerref " << *Mang->getSymbol(F) << "_param_"
          << paramIndex;
      continue;
    }

    if (PAL.hasAttribute(paramIndex + 1, Attribute::ByVal) == false) {
      if (Ty->isVectorTy()) {
        // Just print .param .b8 .align <a> .param[size];
        // <a> = PAL.getparamalignment
        // size = typeallocsize of element type
        unsigned align = PAL.getParamAlignment(paramIndex + 1);
        if (align == 0)
          align = TD->getABITypeAlignment(Ty);

        unsigned sz = TD->getTypeAllocSize(Ty);
        O << "\t.param .align " << align << " .b8 ";
        printParamName(I, paramIndex, O);
        O << "[" << sz << "]";

        continue;
      }
      // Just a scalar
      const PointerType *PTy = dyn_cast<PointerType>(Ty);
      if (isKernelFunc) {
        if (PTy) {
          // Special handling for pointer arguments to kernel
          O << "\t.param .u" << thePointerTy.getSizeInBits() << " ";

          if (nvptxSubtarget.getDrvInterface() != NVPTX::CUDA) {
            Type *ETy = PTy->getElementType();
            int addrSpace = PTy->getAddressSpace();
            switch (addrSpace) {
            default:
              O << ".ptr ";
              break;
            case llvm::ADDRESS_SPACE_CONST_NOT_GEN:
              O << ".ptr .const ";
              break;
            case llvm::ADDRESS_SPACE_SHARED:
              O << ".ptr .shared ";
              break;
            case llvm::ADDRESS_SPACE_GLOBAL:
            case llvm::ADDRESS_SPACE_CONST:
              O << ".ptr .global ";
              break;
            }
            O << ".align " << (int) getOpenCLAlignment(TD, ETy) << " ";
          }
          printParamName(I, paramIndex, O);
          continue;
        }

        // non-pointer scalar to kernel func
        O << "\t.param .";
        // Special case: predicate operands become .u8 types
        if (Ty->isIntegerTy(1))
          O << "u8";
        else
          O << getPTXFundamentalTypeStr(Ty);
        O << " ";
        printParamName(I, paramIndex, O);
        continue;
      }
      // Non-kernel function, just print .param .b<size> for ABI
      // and .reg .b<size> for non ABY
      unsigned sz = 0;
      if (isa<IntegerType>(Ty)) {
        sz = cast<IntegerType>(Ty)->getBitWidth();
        if (sz < 32)
          sz = 32;
      } else if (isa<PointerType>(Ty))
        sz = thePointerTy.getSizeInBits();
      else
        sz = Ty->getPrimitiveSizeInBits();
      if (isABI)
        O << "\t.param .b" << sz << " ";
      else
        O << "\t.reg .b" << sz << " ";
      printParamName(I, paramIndex, O);
      continue;
    }

    // param has byVal attribute. So should be a pointer
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    assert(PTy && "Param with byval attribute should be a pointer type");
    Type *ETy = PTy->getElementType();

    if (isABI || isKernelFunc) {
      // Just print .param .b8 .align <a> .param[size];
      // <a> = PAL.getparamalignment
      // size = typeallocsize of element type
      unsigned align = PAL.getParamAlignment(paramIndex + 1);
      if (align == 0)
        align = TD->getABITypeAlignment(ETy);

      unsigned sz = TD->getTypeAllocSize(ETy);
      O << "\t.param .align " << align << " .b8 ";
      printParamName(I, paramIndex, O);
      O << "[" << sz << "]";
      continue;
    } else {
      // Split the ETy into constituent parts and
      // print .param .b<size> <name> for each part.
      // Further, if a part is vector, print the above for
      // each vector element.
      SmallVector<EVT, 16> vtparts;
      ComputeValueVTs(*TLI, ETy, vtparts);
      for (unsigned i = 0, e = vtparts.size(); i != e; ++i) {
        unsigned elems = 1;
        EVT elemtype = vtparts[i];
        if (vtparts[i].isVector()) {
          elems = vtparts[i].getVectorNumElements();
          elemtype = vtparts[i].getVectorElementType();
        }

        for (unsigned j = 0, je = elems; j != je; ++j) {
          unsigned sz = elemtype.getSizeInBits();
          if (elemtype.isInteger() && (sz < 32))
            sz = 32;
          O << "\t.reg .b" << sz << " ";
          printParamName(I, paramIndex, O);
          if (j < je - 1)
            O << ",\n";
          ++paramIndex;
        }
        if (i < e - 1)
          O << ",\n";
      }
      --paramIndex;
      continue;
    }
  }

  O << "\n)\n";
}

void NVPTXAsmPrinter::emitFunctionParamList(const MachineFunction &MF,
                                            raw_ostream &O) {
  const Function *F = MF.getFunction();
  emitFunctionParamList(F, O);
}

void NVPTXAsmPrinter::setAndEmitFunctionVirtualRegisters(
    const MachineFunction &MF) {
  SmallString<128> Str;
  raw_svector_ostream O(Str);

  // Map the global virtual register number to a register class specific
  // virtual register number starting from 1 with that class.
  const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();
  //unsigned numRegClasses = TRI->getNumRegClasses();

  // Emit the Fake Stack Object
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int NumBytes = (int) MFI->getStackSize();
  if (NumBytes) {
    O << "\t.local .align " << MFI->getMaxAlignment() << " .b8 \t" << DEPOTNAME
      << getFunctionNumber() << "[" << NumBytes << "];\n";
    if (nvptxSubtarget.is64Bit()) {
      O << "\t.reg .b64 \t%SP;\n";
      O << "\t.reg .b64 \t%SPL;\n";
    } else {
      O << "\t.reg .b32 \t%SP;\n";
      O << "\t.reg .b32 \t%SPL;\n";
    }
  }

  // Go through all virtual registers to establish the mapping between the
  // global virtual
  // register number and the per class virtual register number.
  // We use the per class virtual register number in the ptx output.
  unsigned int numVRs = MRI->getNumVirtRegs();
  for (unsigned i = 0; i < numVRs; i++) {
    unsigned int vr = TRI->index2VirtReg(i);
    const TargetRegisterClass *RC = MRI->getRegClass(vr);
    std::map<unsigned, unsigned> &regmap = VRidGlobal2LocalMap[RC->getID()];
    int n = regmap.size();
    regmap.insert(std::make_pair(vr, n + 1));
  }

  // Emit register declarations
  // @TODO: Extract out the real register usage
  O << "\t.reg .pred %p<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .s16 %rc<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .s16 %rs<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .s32 %r<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .s64 %rl<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .f32 %f<" << NVPTXNumRegisters << ">;\n";
  O << "\t.reg .f64 %fl<" << NVPTXNumRegisters << ">;\n";

  // Emit declaration of the virtual registers or 'physical' registers for
  // each register class
  //for (unsigned i=0; i< numRegClasses; i++) {
  //    std::map<unsigned, unsigned> &regmap = VRidGlobal2LocalMap[i];
  //    const TargetRegisterClass *RC = TRI->getRegClass(i);
  //    std::string rcname = getNVPTXRegClassName(RC);
  //    std::string rcStr = getNVPTXRegClassStr(RC);
  //    //int n = regmap.size();
  //    if (!isNVPTXVectorRegClass(RC)) {
  //      O << "\t.reg " << rcname << " \t" << rcStr << "<"
  //        << NVPTXNumRegisters << ">;\n";
  //    }

  // Only declare those registers that may be used. And do not emit vector
  // registers as
  // they are all elementized to scalar registers.
  //if (n && !isNVPTXVectorRegClass(RC)) {
  //    if (RegAllocNilUsed) {
  //        O << "\t.reg " << rcname << " \t" << rcStr << "<" << (n+1)
  //          << ">;\n";
  //    }
  //    else {
  //        O << "\t.reg " << rcname << " \t" << StrToUpper(rcStr)
  //          << "<" << 32 << ">;\n";
  //    }
  //}
  //}

  OutStreamer.EmitRawText(O.str());
}

void NVPTXAsmPrinter::printFPConstant(const ConstantFP *Fp, raw_ostream &O) {
  APFloat APF = APFloat(Fp->getValueAPF()); // make a copy
  bool ignored;
  unsigned int numHex;
  const char *lead;

  if (Fp->getType()->getTypeID() == Type::FloatTyID) {
    numHex = 8;
    lead = "0f";
    APF.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &ignored);
  } else if (Fp->getType()->getTypeID() == Type::DoubleTyID) {
    numHex = 16;
    lead = "0d";
    APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &ignored);
  } else
    llvm_unreachable("unsupported fp type");

  APInt API = APF.bitcastToAPInt();
  std::string hexstr(utohexstr(API.getZExtValue()));
  O << lead;
  if (hexstr.length() < numHex)
    O << std::string(numHex - hexstr.length(), '0');
  O << utohexstr(API.getZExtValue());
}

void NVPTXAsmPrinter::printScalarConstant(const Constant *CPV, raw_ostream &O) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CPV)) {
    O << CI->getValue();
    return;
  }
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CPV)) {
    printFPConstant(CFP, O);
    return;
  }
  if (isa<ConstantPointerNull>(CPV)) {
    O << "0";
    return;
  }
  if (const GlobalValue *GVar = dyn_cast<GlobalValue>(CPV)) {
    O << *Mang->getSymbol(GVar);
    return;
  }
  if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
    const Value *v = Cexpr->stripPointerCasts();
    if (const GlobalValue *GVar = dyn_cast<GlobalValue>(v)) {
      O << *Mang->getSymbol(GVar);
      return;
    } else {
      O << *LowerConstant(CPV, *this);
      return;
    }
  }
  llvm_unreachable("Not scalar type found in printScalarConstant()");
}

void NVPTXAsmPrinter::bufferLEByte(const Constant *CPV, int Bytes,
                                   AggBuffer *aggBuffer) {

  const DataLayout *TD = TM.getDataLayout();

  if (isa<UndefValue>(CPV) || CPV->isNullValue()) {
    int s = TD->getTypeAllocSize(CPV->getType());
    if (s < Bytes)
      s = Bytes;
    aggBuffer->addZeros(s);
    return;
  }

  unsigned char *ptr;
  switch (CPV->getType()->getTypeID()) {

  case Type::IntegerTyID: {
    const Type *ETy = CPV->getType();
    if (ETy == Type::getInt8Ty(CPV->getContext())) {
      unsigned char c =
          (unsigned char)(dyn_cast<ConstantInt>(CPV))->getZExtValue();
      ptr = &c;
      aggBuffer->addBytes(ptr, 1, Bytes);
    } else if (ETy == Type::getInt16Ty(CPV->getContext())) {
      short int16 = (short)(dyn_cast<ConstantInt>(CPV))->getZExtValue();
      ptr = (unsigned char *)&int16;
      aggBuffer->addBytes(ptr, 2, Bytes);
    } else if (ETy == Type::getInt32Ty(CPV->getContext())) {
      if (const ConstantInt *constInt = dyn_cast<ConstantInt>(CPV)) {
        int int32 = (int)(constInt->getZExtValue());
        ptr = (unsigned char *)&int32;
        aggBuffer->addBytes(ptr, 4, Bytes);
        break;
      } else if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
        if (const ConstantInt *constInt = dyn_cast<ConstantInt>(
                ConstantFoldConstantExpression(Cexpr, TD))) {
          int int32 = (int)(constInt->getZExtValue());
          ptr = (unsigned char *)&int32;
          aggBuffer->addBytes(ptr, 4, Bytes);
          break;
        }
        if (Cexpr->getOpcode() == Instruction::PtrToInt) {
          Value *v = Cexpr->getOperand(0)->stripPointerCasts();
          aggBuffer->addSymbol(v);
          aggBuffer->addZeros(4);
          break;
        }
      }
      llvm_unreachable("unsupported integer const type");
    } else if (ETy == Type::getInt64Ty(CPV->getContext())) {
      if (const ConstantInt *constInt = dyn_cast<ConstantInt>(CPV)) {
        long long int64 = (long long)(constInt->getZExtValue());
        ptr = (unsigned char *)&int64;
        aggBuffer->addBytes(ptr, 8, Bytes);
        break;
      } else if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
        if (const ConstantInt *constInt = dyn_cast<ConstantInt>(
                ConstantFoldConstantExpression(Cexpr, TD))) {
          long long int64 = (long long)(constInt->getZExtValue());
          ptr = (unsigned char *)&int64;
          aggBuffer->addBytes(ptr, 8, Bytes);
          break;
        }
        if (Cexpr->getOpcode() == Instruction::PtrToInt) {
          Value *v = Cexpr->getOperand(0)->stripPointerCasts();
          aggBuffer->addSymbol(v);
          aggBuffer->addZeros(8);
          break;
        }
      }
      llvm_unreachable("unsupported integer const type");
    } else
      llvm_unreachable("unsupported integer const type");
    break;
  }
  case Type::FloatTyID:
  case Type::DoubleTyID: {
    const ConstantFP *CFP = dyn_cast<ConstantFP>(CPV);
    const Type *Ty = CFP->getType();
    if (Ty == Type::getFloatTy(CPV->getContext())) {
      float float32 = (float) CFP->getValueAPF().convertToFloat();
      ptr = (unsigned char *)&float32;
      aggBuffer->addBytes(ptr, 4, Bytes);
    } else if (Ty == Type::getDoubleTy(CPV->getContext())) {
      double float64 = CFP->getValueAPF().convertToDouble();
      ptr = (unsigned char *)&float64;
      aggBuffer->addBytes(ptr, 8, Bytes);
    } else {
      llvm_unreachable("unsupported fp const type");
    }
    break;
  }
  case Type::PointerTyID: {
    if (const GlobalValue *GVar = dyn_cast<GlobalValue>(CPV)) {
      aggBuffer->addSymbol(GVar);
    } else if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
      const Value *v = Cexpr->stripPointerCasts();
      aggBuffer->addSymbol(v);
    }
    unsigned int s = TD->getTypeAllocSize(CPV->getType());
    aggBuffer->addZeros(s);
    break;
  }

  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::StructTyID: {
    if (isa<ConstantArray>(CPV) || isa<ConstantVector>(CPV) ||
        isa<ConstantStruct>(CPV)) {
      int ElementSize = TD->getTypeAllocSize(CPV->getType());
      bufferAggregateConstant(CPV, aggBuffer);
      if (Bytes > ElementSize)
        aggBuffer->addZeros(Bytes - ElementSize);
    } else if (isa<ConstantAggregateZero>(CPV))
      aggBuffer->addZeros(Bytes);
    else
      llvm_unreachable("Unexpected Constant type");
    break;
  }

  default:
    llvm_unreachable("unsupported type");
  }
}

void NVPTXAsmPrinter::bufferAggregateConstant(const Constant *CPV,
                                              AggBuffer *aggBuffer) {
  const DataLayout *TD = TM.getDataLayout();
  int Bytes;

  // Old constants
  if (isa<ConstantArray>(CPV) || isa<ConstantVector>(CPV)) {
    if (CPV->getNumOperands())
      for (unsigned i = 0, e = CPV->getNumOperands(); i != e; ++i)
        bufferLEByte(cast<Constant>(CPV->getOperand(i)), 0, aggBuffer);
    return;
  }

  if (const ConstantDataSequential *CDS =
          dyn_cast<ConstantDataSequential>(CPV)) {
    if (CDS->getNumElements())
      for (unsigned i = 0; i < CDS->getNumElements(); ++i)
        bufferLEByte(cast<Constant>(CDS->getElementAsConstant(i)), 0,
                     aggBuffer);
    return;
  }

  if (isa<ConstantStruct>(CPV)) {
    if (CPV->getNumOperands()) {
      StructType *ST = cast<StructType>(CPV->getType());
      for (unsigned i = 0, e = CPV->getNumOperands(); i != e; ++i) {
        if (i == (e - 1))
          Bytes = TD->getStructLayout(ST)->getElementOffset(0) +
                  TD->getTypeAllocSize(ST) -
                  TD->getStructLayout(ST)->getElementOffset(i);
        else
          Bytes = TD->getStructLayout(ST)->getElementOffset(i + 1) -
                  TD->getStructLayout(ST)->getElementOffset(i);
        bufferLEByte(cast<Constant>(CPV->getOperand(i)), Bytes, aggBuffer);
      }
    }
    return;
  }
  llvm_unreachable("unsupported constant type in printAggregateConstant()");
}

// buildTypeNameMap - Run through symbol table looking for type names.
//

bool NVPTXAsmPrinter::isImageType(const Type *Ty) {

  std::map<const Type *, std::string>::iterator PI = TypeNameMap.find(Ty);

  if (PI != TypeNameMap.end() && (!PI->second.compare("struct._image1d_t") ||
                                  !PI->second.compare("struct._image2d_t") ||
                                  !PI->second.compare("struct._image3d_t")))
    return true;

  return false;
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool NVPTXAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      unsigned AsmVariant,
                                      const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    case 'r':
      break;
    }
  }

  printOperand(MI, OpNo, O);

  return false;
}

bool NVPTXAsmPrinter::PrintAsmMemoryOperand(
    const MachineInstr *MI, unsigned OpNo, unsigned AsmVariant,
    const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier

  O << '[';
  printMemOperand(MI, OpNo, O);
  O << ']';

  return false;
}

bool NVPTXAsmPrinter::ignoreLoc(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case NVPTX::CallArgBeginInst:
  case NVPTX::CallArgEndInst0:
  case NVPTX::CallArgEndInst1:
  case NVPTX::CallArgF32:
  case NVPTX::CallArgF64:
  case NVPTX::CallArgI16:
  case NVPTX::CallArgI32:
  case NVPTX::CallArgI32imm:
  case NVPTX::CallArgI64:
  case NVPTX::CallArgI8:
  case NVPTX::CallArgParam:
  case NVPTX::CallVoidInst:
  case NVPTX::CallVoidInstReg:
  case NVPTX::Callseq_End:
  case NVPTX::CallVoidInstReg64:
  case NVPTX::DeclareParamInst:
  case NVPTX::DeclareRetMemInst:
  case NVPTX::DeclareRetRegInst:
  case NVPTX::DeclareRetScalarInst:
  case NVPTX::DeclareScalarParamInst:
  case NVPTX::DeclareScalarRegInst:
  case NVPTX::StoreParamF32:
  case NVPTX::StoreParamF64:
  case NVPTX::StoreParamI16:
  case NVPTX::StoreParamI32:
  case NVPTX::StoreParamI64:
  case NVPTX::StoreParamI8:
  case NVPTX::StoreParamS32I8:
  case NVPTX::StoreParamU32I8:
  case NVPTX::StoreParamS32I16:
  case NVPTX::StoreParamU32I16:
  case NVPTX::StoreRetvalF32:
  case NVPTX::StoreRetvalF64:
  case NVPTX::StoreRetvalI16:
  case NVPTX::StoreRetvalI32:
  case NVPTX::StoreRetvalI64:
  case NVPTX::StoreRetvalI8:
  case NVPTX::LastCallArgF32:
  case NVPTX::LastCallArgF64:
  case NVPTX::LastCallArgI16:
  case NVPTX::LastCallArgI32:
  case NVPTX::LastCallArgI32imm:
  case NVPTX::LastCallArgI64:
  case NVPTX::LastCallArgI8:
  case NVPTX::LastCallArgParam:
  case NVPTX::LoadParamMemF32:
  case NVPTX::LoadParamMemF64:
  case NVPTX::LoadParamMemI16:
  case NVPTX::LoadParamMemI32:
  case NVPTX::LoadParamMemI64:
  case NVPTX::LoadParamMemI8:
  case NVPTX::LoadParamRegF32:
  case NVPTX::LoadParamRegF64:
  case NVPTX::LoadParamRegI16:
  case NVPTX::LoadParamRegI32:
  case NVPTX::LoadParamRegI64:
  case NVPTX::LoadParamRegI8:
  case NVPTX::PrototypeInst:
  case NVPTX::DBG_VALUE:
    return true;
  }
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeNVPTXBackendAsmPrinter() {
  RegisterAsmPrinter<NVPTXAsmPrinter> X(TheNVPTXTarget32);
  RegisterAsmPrinter<NVPTXAsmPrinter> Y(TheNVPTXTarget64);
}

void NVPTXAsmPrinter::emitSrcInText(StringRef filename, unsigned line) {
  std::stringstream temp;
  LineReader *reader = this->getReader(filename.str());
  temp << "\n//";
  temp << filename.str();
  temp << ":";
  temp << line;
  temp << " ";
  temp << reader->readLine(line);
  temp << "\n";
  this->OutStreamer.EmitRawText(Twine(temp.str()));
}

LineReader *NVPTXAsmPrinter::getReader(std::string filename) {
  if (reader == NULL) {
    reader = new LineReader(filename);
  }

  if (reader->fileName() != filename) {
    delete reader;
    reader = new LineReader(filename);
  }

  return reader;
}

std::string LineReader::readLine(unsigned lineNum) {
  if (lineNum < theCurLine) {
    theCurLine = 0;
    fstr.seekg(0, std::ios::beg);
  }
  while (theCurLine < lineNum) {
    fstr.getline(buff, 500);
    theCurLine++;
  }
  return buff;
}

// Force static initialization.
extern "C" void LLVMInitializeNVPTXAsmPrinter() {
  RegisterAsmPrinter<NVPTXAsmPrinter> X(TheNVPTXTarget32);
  RegisterAsmPrinter<NVPTXAsmPrinter> Y(TheNVPTXTarget64);
}
