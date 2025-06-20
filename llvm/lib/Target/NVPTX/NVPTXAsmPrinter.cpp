//===-- NVPTXAsmPrinter.cpp - NVPTX LLVM assembly writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to NVPTX assembly language.
//
//===----------------------------------------------------------------------===//

#include "NVPTXAsmPrinter.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "MCTargetDesc/NVPTXInstPrinter.h"
#include "MCTargetDesc/NVPTXMCAsmInfo.h"
#include "MCTargetDesc/NVPTXTargetStreamer.h"
#include "NVPTX.h"
#include "NVPTXMCExpr.h"
#include "NVPTXMachineFunctionInfo.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "TargetInfo/NVPTXTargetInfo.h"
#include "cl_common_defines.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEPOTNAME "__local_depot"

/// discoverDependentGlobals - Return a set of GlobalVariables on which \p V
/// depends.
static void
discoverDependentGlobals(const Value *V,
                         DenseSet<const GlobalVariable *> &Globals) {
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    Globals.insert(GV);
    return;
  }

  if (const User *U = dyn_cast<User>(V))
    for (const auto &O : U->operands())
      discoverDependentGlobals(O, Globals);
}

/// VisitGlobalVariableForEmission - Add \p GV to the list of GlobalVariable
/// instances to be emitted, but only after any dependents have been added
/// first.s
static void
VisitGlobalVariableForEmission(const GlobalVariable *GV,
                               SmallVectorImpl<const GlobalVariable *> &Order,
                               DenseSet<const GlobalVariable *> &Visited,
                               DenseSet<const GlobalVariable *> &Visiting) {
  // Have we already visited this one?
  if (Visited.count(GV))
    return;

  // Do we have a circular dependency?
  if (!Visiting.insert(GV).second)
    report_fatal_error("Circular dependency found in global variable set");

  // Make sure we visit all dependents first
  DenseSet<const GlobalVariable *> Others;
  for (const auto &O : GV->operands())
    discoverDependentGlobals(O, Others);

  for (const GlobalVariable *GV : Others)
    VisitGlobalVariableForEmission(GV, Order, Visited, Visiting);

  // Now we can visit ourself
  Order.push_back(GV);
  Visited.insert(GV);
  Visiting.erase(GV);
}

void NVPTXAsmPrinter::emitInstruction(const MachineInstr *MI) {
  NVPTX_MC::verifyInstructionPredicates(MI->getOpcode(),
                                        getSubtargetInfo().getFeatureBits());

  MCInst Inst;
  lowerToMCInst(MI, Inst);
  EmitToStreamer(*OutStreamer, Inst);
}

void NVPTXAsmPrinter::lowerToMCInst(const MachineInstr *MI, MCInst &OutMI) {
  OutMI.setOpcode(MI->getOpcode());
  // Special: Do not mangle symbol operand of CALL_PROTOTYPE
  if (MI->getOpcode() == NVPTX::CALL_PROTOTYPE) {
    const MachineOperand &MO = MI->getOperand(0);
    OutMI.addOperand(GetSymbolRef(
      OutContext.getOrCreateSymbol(Twine(MO.getSymbolName()))));
    return;
  }

  for (const auto MO : MI->operands())
    OutMI.addOperand(lowerOperand(MO));
}

MCOperand NVPTXAsmPrinter::lowerOperand(const MachineOperand &MO) {
  switch (MO.getType()) {
  default:
    llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    return MCOperand::createReg(encodeVirtualRegister(MO.getReg()));
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());
  case MachineOperand::MO_MachineBasicBlock:
    return MCOperand::createExpr(
        MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), OutContext));
  case MachineOperand::MO_ExternalSymbol:
    return GetSymbolRef(GetExternalSymbolSymbol(MO.getSymbolName()));
  case MachineOperand::MO_GlobalAddress:
    return GetSymbolRef(getSymbol(MO.getGlobal()));
  case MachineOperand::MO_FPImmediate: {
    const ConstantFP *Cnt = MO.getFPImm();
    const APFloat &Val = Cnt->getValueAPF();

    switch (Cnt->getType()->getTypeID()) {
    default:
      report_fatal_error("Unsupported FP type");
      break;
    case Type::HalfTyID:
      return MCOperand::createExpr(
          NVPTXFloatMCExpr::createConstantFPHalf(Val, OutContext));
    case Type::BFloatTyID:
      return MCOperand::createExpr(
          NVPTXFloatMCExpr::createConstantBFPHalf(Val, OutContext));
    case Type::FloatTyID:
      return MCOperand::createExpr(
          NVPTXFloatMCExpr::createConstantFPSingle(Val, OutContext));
    case Type::DoubleTyID:
      return MCOperand::createExpr(
          NVPTXFloatMCExpr::createConstantFPDouble(Val, OutContext));
    }
    break;
  }
  }
}

unsigned NVPTXAsmPrinter::encodeVirtualRegister(unsigned Reg) {
  if (Register::isVirtualRegister(Reg)) {
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);

    DenseMap<unsigned, unsigned> &RegMap = VRegMapping[RC];
    unsigned RegNum = RegMap[Reg];

    // Encode the register class in the upper 4 bits
    // Must be kept in sync with NVPTXInstPrinter::printRegName
    unsigned Ret = 0;
    if (RC == &NVPTX::Int1RegsRegClass) {
      Ret = (1 << 28);
    } else if (RC == &NVPTX::Int16RegsRegClass) {
      Ret = (2 << 28);
    } else if (RC == &NVPTX::Int32RegsRegClass) {
      Ret = (3 << 28);
    } else if (RC == &NVPTX::Int64RegsRegClass) {
      Ret = (4 << 28);
    } else if (RC == &NVPTX::Int128RegsRegClass) {
      Ret = (7 << 28);
    } else {
      report_fatal_error("Bad register class");
    }

    // Insert the vreg number
    Ret |= (RegNum & 0x0FFFFFFF);
    return Ret;
  } else {
    // Some special-use registers are actually physical registers.
    // Encode this as the register class ID of 0 and the real register ID.
    return Reg & 0x0FFFFFFF;
  }
}

MCOperand NVPTXAsmPrinter::GetSymbolRef(const MCSymbol *Symbol) {
  const MCExpr *Expr;
  Expr = MCSymbolRefExpr::create(Symbol, OutContext);
  return MCOperand::createExpr(Expr);
}

void NVPTXAsmPrinter::printReturnValStr(const Function *F, raw_ostream &O) {
  const DataLayout &DL = getDataLayout();
  const NVPTXSubtarget &STI = TM.getSubtarget<NVPTXSubtarget>(*F);
  const auto *TLI = cast<NVPTXTargetLowering>(STI.getTargetLowering());

  Type *Ty = F->getReturnType();
  if (Ty->getTypeID() == Type::VoidTyID)
    return;
  O << " (";

  auto PrintScalarRetVal = [&](unsigned Size) {
    O << ".param .b" << promoteScalarArgumentSize(Size) << " func_retval0";
  };
  if (shouldPassAsArray(Ty)) {
    const unsigned TotalSize = DL.getTypeAllocSize(Ty);
    const Align RetAlignment = TLI->getFunctionArgumentAlignment(
        F, Ty, AttributeList::ReturnIndex, DL);
    O << ".param .align " << RetAlignment.value() << " .b8 func_retval0["
      << TotalSize << "]";
  } else if (Ty->isFloatingPointTy()) {
    PrintScalarRetVal(Ty->getPrimitiveSizeInBits());
  } else if (auto *ITy = dyn_cast<IntegerType>(Ty)) {
    PrintScalarRetVal(ITy->getBitWidth());
  } else if (isa<PointerType>(Ty)) {
    PrintScalarRetVal(TLI->getPointerTy(DL).getSizeInBits());
  } else
    llvm_unreachable("Unknown return type");
  O << ") ";
}

void NVPTXAsmPrinter::printReturnValStr(const MachineFunction &MF,
                                        raw_ostream &O) {
  const Function &F = MF.getFunction();
  printReturnValStr(&F, O);
}

// Return true if MBB is the header of a loop marked with
// llvm.loop.unroll.disable or llvm.loop.unroll.count=1.
bool NVPTXAsmPrinter::isLoopHeaderOfNoUnroll(
    const MachineBasicBlock &MBB) const {
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  // We insert .pragma "nounroll" only to the loop header.
  if (!LI.isLoopHeader(&MBB))
    return false;

  // llvm.loop.unroll.disable is marked on the back edges of a loop. Therefore,
  // we iterate through each back edge of the loop with header MBB, and check
  // whether its metadata contains llvm.loop.unroll.disable.
  for (const MachineBasicBlock *PMBB : MBB.predecessors()) {
    if (LI.getLoopFor(PMBB) != LI.getLoopFor(&MBB)) {
      // Edges from other loops to MBB are not back edges.
      continue;
    }
    if (const BasicBlock *PBB = PMBB->getBasicBlock()) {
      if (MDNode *LoopID =
              PBB->getTerminator()->getMetadata(LLVMContext::MD_loop)) {
        if (GetUnrollMetadata(LoopID, "llvm.loop.unroll.disable"))
          return true;
        if (MDNode *UnrollCountMD =
                GetUnrollMetadata(LoopID, "llvm.loop.unroll.count")) {
          if (mdconst::extract<ConstantInt>(UnrollCountMD->getOperand(1))
                  ->isOne())
            return true;
        }
      }
    }
  }
  return false;
}

void NVPTXAsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  AsmPrinter::emitBasicBlockStart(MBB);
  if (isLoopHeaderOfNoUnroll(MBB))
    OutStreamer->emitRawText(StringRef("\t.pragma \"nounroll\";\n"));
}

void NVPTXAsmPrinter::emitFunctionEntryLabel() {
  SmallString<128> Str;
  raw_svector_ostream O(Str);

  if (!GlobalsEmitted) {
    emitGlobals(*MF->getFunction().getParent());
    GlobalsEmitted = true;
  }

  // Set up
  MRI = &MF->getRegInfo();
  F = &MF->getFunction();
  emitLinkageDirective(F, O);
  if (isKernelFunction(*F))
    O << ".entry ";
  else {
    O << ".func ";
    printReturnValStr(*MF, O);
  }

  CurrentFnSym->print(O, MAI);

  emitFunctionParamList(F, O);
  O << "\n";

  if (isKernelFunction(*F))
    emitKernelFunctionDirectives(*F, O);

  if (shouldEmitPTXNoReturn(F, TM))
    O << ".noreturn";

  OutStreamer->emitRawText(O.str());

  VRegMapping.clear();
  // Emit open brace for function body.
  OutStreamer->emitRawText(StringRef("{\n"));
  setAndEmitFunctionVirtualRegisters(*MF);
  encodeDebugInfoRegisterNumbers(*MF);
  // Emit initial .loc debug directive for correct relocation symbol data.
  if (const DISubprogram *SP = MF->getFunction().getSubprogram()) {
    assert(SP->getUnit());
    if (!SP->getUnit()->isDebugDirectivesOnly())
      emitInitialRawDwarfLocDirective(*MF);
  }
}

bool NVPTXAsmPrinter::runOnMachineFunction(MachineFunction &F) {
  bool Result = AsmPrinter::runOnMachineFunction(F);
  // Emit closing brace for the body of function F.
  // The closing brace must be emitted here because we need to emit additional
  // debug labels/data after the last basic block.
  // We need to emit the closing brace here because we don't have function that
  // finished emission of the function body.
  OutStreamer->emitRawText(StringRef("}\n"));
  return Result;
}

void NVPTXAsmPrinter::emitFunctionBodyStart() {
  SmallString<128> Str;
  raw_svector_ostream O(Str);
  emitDemotedVars(&MF->getFunction(), O);
  OutStreamer->emitRawText(O.str());
}

void NVPTXAsmPrinter::emitFunctionBodyEnd() {
  VRegMapping.clear();
}

const MCSymbol *NVPTXAsmPrinter::getFunctionFrameSymbol() const {
    SmallString<128> Str;
    raw_svector_ostream(Str) << DEPOTNAME << getFunctionNumber();
    return OutContext.getOrCreateSymbol(Str);
}

void NVPTXAsmPrinter::emitImplicitDef(const MachineInstr *MI) const {
  Register RegNo = MI->getOperand(0).getReg();
  if (RegNo.isVirtual()) {
    OutStreamer->AddComment(Twine("implicit-def: ") +
                            getVirtualRegisterName(RegNo));
  } else {
    const NVPTXSubtarget &STI = MI->getMF()->getSubtarget<NVPTXSubtarget>();
    OutStreamer->AddComment(Twine("implicit-def: ") +
                            STI.getRegisterInfo()->getName(RegNo));
  }
  OutStreamer->addBlankLine();
}

void NVPTXAsmPrinter::emitKernelFunctionDirectives(const Function &F,
                                                   raw_ostream &O) const {
  // If the NVVM IR has some of reqntid* specified, then output
  // the reqntid directive, and set the unspecified ones to 1.
  // If none of Reqntid* is specified, don't output reqntid directive.
  const auto ReqNTID = getReqNTID(F);
  if (!ReqNTID.empty())
    O << formatv(".reqntid {0:$[, ]}\n",
                 make_range(ReqNTID.begin(), ReqNTID.end()));

  const auto MaxNTID = getMaxNTID(F);
  if (!MaxNTID.empty())
    O << formatv(".maxntid {0:$[, ]}\n",
                 make_range(MaxNTID.begin(), MaxNTID.end()));

  if (const auto Mincta = getMinCTASm(F))
    O << ".minnctapersm " << *Mincta << "\n";

  if (const auto Maxnreg = getMaxNReg(F))
    O << ".maxnreg " << *Maxnreg << "\n";

  // .maxclusterrank directive requires SM_90 or higher, make sure that we
  // filter it out for lower SM versions, as it causes a hard ptxas crash.
  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  const auto *STI = static_cast<const NVPTXSubtarget *>(NTM.getSubtargetImpl());

  if (STI->getSmVersion() >= 90) {
    const auto ClusterDim = getClusterDim(F);

    if (!ClusterDim.empty()) {
      O << ".explicitcluster\n";
      if (ClusterDim[0] != 0) {
        assert(llvm::all_of(ClusterDim, [](unsigned D) { return D != 0; }) &&
               "cluster_dim_x != 0 implies cluster_dim_y and cluster_dim_z "
               "should be non-zero as well");

        O << formatv(".reqnctapercluster {0:$[, ]}\n",
                     make_range(ClusterDim.begin(), ClusterDim.end()));
      } else {
        assert(llvm::all_of(ClusterDim, [](unsigned D) { return D == 0; }) &&
               "cluster_dim_x == 0 implies cluster_dim_y and cluster_dim_z "
               "should be 0 as well");
      }
    }
    if (const auto Maxclusterrank = getMaxClusterRank(F))
      O << ".maxclusterrank " << *Maxclusterrank << "\n";
  }
}

std::string NVPTXAsmPrinter::getVirtualRegisterName(unsigned Reg) const {
  const TargetRegisterClass *RC = MRI->getRegClass(Reg);

  std::string Name;
  raw_string_ostream NameStr(Name);

  VRegRCMap::const_iterator I = VRegMapping.find(RC);
  assert(I != VRegMapping.end() && "Bad register class");
  const DenseMap<unsigned, unsigned> &RegMap = I->second;

  VRegMap::const_iterator VI = RegMap.find(Reg);
  assert(VI != RegMap.end() && "Bad virtual register");
  unsigned MappedVR = VI->second;

  NameStr << getNVPTXRegClassStr(RC) << MappedVR;

  return Name;
}

void NVPTXAsmPrinter::emitVirtualRegister(unsigned int vr,
                                          raw_ostream &O) {
  O << getVirtualRegisterName(vr);
}

void NVPTXAsmPrinter::emitAliasDeclaration(const GlobalAlias *GA,
                                           raw_ostream &O) {
  const Function *F = dyn_cast_or_null<Function>(GA->getAliaseeObject());
  if (!F || isKernelFunction(*F) || F->isDeclaration())
    report_fatal_error(
        "NVPTX aliasee must be a non-kernel function definition");

  if (GA->hasLinkOnceLinkage() || GA->hasWeakLinkage() ||
      GA->hasAvailableExternallyLinkage() || GA->hasCommonLinkage())
    report_fatal_error("NVPTX aliasee must not be '.weak'");

  emitDeclarationWithName(F, getSymbol(GA), O);
}

void NVPTXAsmPrinter::emitDeclaration(const Function *F, raw_ostream &O) {
  emitDeclarationWithName(F, getSymbol(F), O);
}

void NVPTXAsmPrinter::emitDeclarationWithName(const Function *F, MCSymbol *S,
                                              raw_ostream &O) {
  emitLinkageDirective(F, O);
  if (isKernelFunction(*F))
    O << ".entry ";
  else
    O << ".func ";
  printReturnValStr(F, O);
  S->print(O, MAI);
  O << "\n";
  emitFunctionParamList(F, O);
  O << "\n";
  if (shouldEmitPTXNoReturn(F, TM))
    O << ".noreturn";
  O << ";\n";
}

static bool usedInGlobalVarDef(const Constant *C) {
  if (!C)
    return false;

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    return GV->getName() != "llvm.used";

  for (const User *U : C->users())
    if (const Constant *C = dyn_cast<Constant>(U))
      if (usedInGlobalVarDef(C))
        return true;

  return false;
}

static bool usedInOneFunc(const User *U, Function const *&OneFunc) {
  if (const GlobalVariable *OtherGV = dyn_cast<GlobalVariable>(U))
    if (OtherGV->getName() == "llvm.used")
      return true;

  if (const Instruction *I = dyn_cast<Instruction>(U)) {
    if (const Function *CurFunc = I->getFunction()) {
      if (OneFunc && (CurFunc != OneFunc))
        return false;
      OneFunc = CurFunc;
      return true;
    }
    return false;
  }

  for (const User *UU : U->users())
    if (!usedInOneFunc(UU, OneFunc))
      return false;

  return true;
}

/* Find out if a global variable can be demoted to local scope.
 * Currently, this is valid for CUDA shared variables, which have local
 * scope and global lifetime. So the conditions to check are :
 * 1. Is the global variable in shared address space?
 * 2. Does it have local linkage?
 * 3. Is the global variable referenced only in one function?
 */
static bool canDemoteGlobalVar(const GlobalVariable *GV, Function const *&f) {
  if (!GV->hasLocalLinkage())
    return false;
  if (GV->getAddressSpace() != ADDRESS_SPACE_SHARED)
    return false;

  const Function *oneFunc = nullptr;

  bool flag = usedInOneFunc(GV, oneFunc);
  if (!flag)
    return false;
  if (!oneFunc)
    return false;
  f = oneFunc;
  return true;
}

static bool useFuncSeen(const Constant *C,
                        const SmallPtrSetImpl<const Function *> &SeenSet) {
  for (const User *U : C->users()) {
    if (const Constant *cu = dyn_cast<Constant>(U)) {
      if (useFuncSeen(cu, SeenSet))
        return true;
    } else if (const Instruction *I = dyn_cast<Instruction>(U)) {
      if (const Function *Caller = I->getFunction())
        if (SeenSet.contains(Caller))
          return true;
    }
  }
  return false;
}

void NVPTXAsmPrinter::emitDeclarations(const Module &M, raw_ostream &O) {
  SmallPtrSet<const Function *, 32> SeenSet;
  for (const Function &F : M) {
    if (F.getAttributes().hasFnAttr("nvptx-libcall-callee")) {
      emitDeclaration(&F, O);
      continue;
    }

    if (F.isDeclaration()) {
      if (F.use_empty())
        continue;
      if (F.getIntrinsicID())
        continue;
      emitDeclaration(&F, O);
      continue;
    }
    for (const User *U : F.users()) {
      if (const Constant *C = dyn_cast<Constant>(U)) {
        if (usedInGlobalVarDef(C)) {
          // The use is in the initialization of a global variable
          // that is a function pointer, so print a declaration
          // for the original function
          emitDeclaration(&F, O);
          break;
        }
        // Emit a declaration of this function if the function that
        // uses this constant expr has already been seen.
        if (useFuncSeen(C, SeenSet)) {
          emitDeclaration(&F, O);
          break;
        }
      }

      if (!isa<Instruction>(U))
        continue;
      const Function *Caller = cast<Instruction>(U)->getFunction();
      if (!Caller)
        continue;

      // If a caller has already been seen, then the caller is
      // appearing in the module before the callee. so print out
      // a declaration for the callee.
      if (SeenSet.contains(Caller)) {
        emitDeclaration(&F, O);
        break;
      }
    }
    SeenSet.insert(&F);
  }
  for (const GlobalAlias &GA : M.aliases())
    emitAliasDeclaration(&GA, O);
}

void NVPTXAsmPrinter::emitStartOfAsmFile(Module &M) {
  // Construct a default subtarget off of the TargetMachine defaults. The
  // rest of NVPTX isn't friendly to change subtargets per function and
  // so the default TargetMachine will have all of the options.
  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  const auto* STI = static_cast<const NVPTXSubtarget*>(NTM.getSubtargetImpl());
  SmallString<128> Str1;
  raw_svector_ostream OS1(Str1);

  // Emit header before any dwarf directives are emitted below.
  emitHeader(M, OS1, *STI);
  OutStreamer->emitRawText(OS1.str());
}

bool NVPTXAsmPrinter::doInitialization(Module &M) {
  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  const NVPTXSubtarget &STI =
      *static_cast<const NVPTXSubtarget *>(NTM.getSubtargetImpl());
  if (M.alias_size() && (STI.getPTXVersion() < 63 || STI.getSmVersion() < 30))
    report_fatal_error(".alias requires PTX version >= 6.3 and sm_30");

  // We need to call the parent's one explicitly.
  bool Result = AsmPrinter::doInitialization(M);

  GlobalsEmitted = false;

  return Result;
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
  for (const GlobalVariable &I : M.globals())
    VisitGlobalVariableForEmission(&I, Globals, GVVisited, GVVisiting);

  assert(GVVisited.size() == M.global_size() && "Missed a global variable");
  assert(GVVisiting.size() == 0 && "Did not fully process a global variable");

  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  const NVPTXSubtarget &STI =
      *static_cast<const NVPTXSubtarget *>(NTM.getSubtargetImpl());

  // Print out module-level global variables in proper order
  for (const GlobalVariable *GV : Globals)
    printModuleLevelGV(GV, OS2, /*ProcessDemoted=*/false, STI);

  OS2 << '\n';

  OutStreamer->emitRawText(OS2.str());
}

void NVPTXAsmPrinter::emitGlobalAlias(const Module &M, const GlobalAlias &GA) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  MCSymbol *Name = getSymbol(&GA);

  OS << ".alias " << Name->getName() << ", " << GA.getAliaseeObject()->getName()
     << ";\n";

  OutStreamer->emitRawText(OS.str());
}

void NVPTXAsmPrinter::emitHeader(Module &M, raw_ostream &O,
                                 const NVPTXSubtarget &STI) {
  const unsigned PTXVersion = STI.getPTXVersion();

  O << "//\n"
       "// Generated by LLVM NVPTX Back-End\n"
       "//\n"
       "\n"
    << ".version " << (PTXVersion / 10) << "." << (PTXVersion % 10) << "\n"
    << ".target " << STI.getTargetName();

  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  if (NTM.getDrvInterface() == NVPTX::NVCL)
    O << ", texmode_independent";

  bool HasFullDebugInfo = false;
  for (DICompileUnit *CU : M.debug_compile_units()) {
    switch(CU->getEmissionKind()) {
    case DICompileUnit::NoDebug:
    case DICompileUnit::DebugDirectivesOnly:
      break;
    case DICompileUnit::LineTablesOnly:
    case DICompileUnit::FullDebug:
      HasFullDebugInfo = true;
      break;
    }
    if (HasFullDebugInfo)
      break;
  }
  if (HasFullDebugInfo)
    O << ", debug";

  O << "\n"
    << ".address_size " << (NTM.is64Bit() ? "64" : "32") << "\n"
    << "\n";
}

bool NVPTXAsmPrinter::doFinalization(Module &M) {
  // If we did not emit any functions, then the global declarations have not
  // yet been emitted.
  if (!GlobalsEmitted) {
    emitGlobals(M);
    GlobalsEmitted = true;
  }

  // call doFinalization
  bool ret = AsmPrinter::doFinalization(M);

  clearAnnotationCache(&M);

  auto *TS =
      static_cast<NVPTXTargetStreamer *>(OutStreamer->getTargetStreamer());
  // Close the last emitted section
  if (hasDebugInfo()) {
    TS->closeLastSection();
    // Emit empty .debug_macinfo section for better support of the empty files.
    OutStreamer->emitRawText("\t.section\t.debug_macinfo\t{\t}");
  }

  // Output last DWARF .file directives, if any.
  TS->outputDwarfFileDirectives();

  return ret;
}

// This function emits appropriate linkage directives for
// functions and global variables.
//
// extern function declaration            -> .extern
// extern function definition             -> .visible
// external global variable with init     -> .visible
// external without init                  -> .extern
// appending                              -> not allowed, assert.
// for any linkage other than
// internal, private, linker_private,
// linker_private_weak, linker_private_weak_def_auto,
// we emit                                -> .weak.

void NVPTXAsmPrinter::emitLinkageDirective(const GlobalValue *V,
                                           raw_ostream &O) {
  if (static_cast<NVPTXTargetMachine &>(TM).getDrvInterface() == NVPTX::CUDA) {
    if (V->hasExternalLinkage()) {
      if (const auto *GVar = dyn_cast<GlobalVariable>(V))
        O << (GVar->hasInitializer() ? ".visible " : ".extern ");
      else if (V->isDeclaration())
        O << ".extern ";
      else
        O << ".visible ";
    } else if (V->hasAppendingLinkage()) {
      report_fatal_error("Symbol '" + (V->hasName() ? V->getName() : "") +
                         "' has unsupported appending linkage type");
    } else if (!V->hasInternalLinkage() && !V->hasPrivateLinkage()) {
      O << ".weak ";
    }
  }
}

void NVPTXAsmPrinter::printModuleLevelGV(const GlobalVariable *GVar,
                                         raw_ostream &O, bool ProcessDemoted,
                                         const NVPTXSubtarget &STI) {
  // Skip meta data
  if (GVar->hasSection())
    if (GVar->getSection() == "llvm.metadata")
      return;

  // Skip LLVM intrinsic global variables
  if (GVar->getName().starts_with("llvm.") ||
      GVar->getName().starts_with("nvvm."))
    return;

  const DataLayout &DL = getDataLayout();

  // GlobalVariables are always constant pointers themselves.
  Type *ETy = GVar->getValueType();

  if (GVar->hasExternalLinkage()) {
    if (GVar->hasInitializer())
      O << ".visible ";
    else
      O << ".extern ";
  } else if (STI.getPTXVersion() >= 50 && GVar->hasCommonLinkage() &&
             GVar->getAddressSpace() == ADDRESS_SPACE_GLOBAL) {
    O << ".common ";
  } else if (GVar->hasLinkOnceLinkage() || GVar->hasWeakLinkage() ||
             GVar->hasAvailableExternallyLinkage() ||
             GVar->hasCommonLinkage()) {
    O << ".weak ";
  }

  if (isTexture(*GVar)) {
    O << ".global .texref " << getTextureName(*GVar) << ";\n";
    return;
  }

  if (isSurface(*GVar)) {
    O << ".global .surfref " << getSurfaceName(*GVar) << ";\n";
    return;
  }

  if (GVar->isDeclaration()) {
    // (extern) declarations, no definition or initializer
    // Currently the only known declaration is for an automatic __local
    // (.shared) promoted to global.
    emitPTXGlobalVariable(GVar, O, STI);
    O << ";\n";
    return;
  }

  if (isSampler(*GVar)) {
    O << ".global .samplerref " << getSamplerName(*GVar);

    const Constant *Initializer = nullptr;
    if (GVar->hasInitializer())
      Initializer = GVar->getInitializer();
    const ConstantInt *CI = nullptr;
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
        llvm_unreachable("Anisotropic filtering is not supported");
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
    if (GVar->getName().starts_with("unrollpragma"))
      return;

    // FIXME - need better way (e.g. Metadata) to avoid generating this global
    if (GVar->getName().starts_with("filename"))
      return;
    if (GVar->use_empty())
      return;
  }

  const Function *DemotedFunc = nullptr;
  if (!ProcessDemoted && canDemoteGlobalVar(GVar, DemotedFunc)) {
    O << "// " << GVar->getName() << " has been demoted\n";
    localDecls[DemotedFunc].push_back(GVar);
    return;
  }

  O << ".";
  emitPTXAddressSpace(GVar->getAddressSpace(), O);

  if (isManaged(*GVar)) {
    if (STI.getPTXVersion() < 40 || STI.getSmVersion() < 30)
      report_fatal_error(
          ".attribute(.managed) requires PTX version >= 4.0 and sm_30");
    O << " .attribute(.managed)";
  }

  O << " .align "
    << GVar->getAlign().value_or(DL.getPrefTypeAlign(ETy)).value();

  if (ETy->isPointerTy() || ((ETy->isIntegerTy() || ETy->isFloatingPointTy()) &&
                             ETy->getScalarSizeInBits() <= 64)) {
    O << " .";
    // Special case: ABI requires that we use .u8 for predicates
    if (ETy->isIntegerTy(1))
      O << "u8";
    else
      O << getPTXFundamentalTypeStr(ETy, false);
    O << " ";
    getSymbol(GVar)->print(O, MAI);

    // Ptx allows variable initilization only for constant and global state
    // spaces.
    if (GVar->hasInitializer()) {
      if ((GVar->getAddressSpace() == ADDRESS_SPACE_GLOBAL) ||
          (GVar->getAddressSpace() == ADDRESS_SPACE_CONST)) {
        const Constant *Initializer = GVar->getInitializer();
        // 'undef' is treated as there is no value specified.
        if (!Initializer->isNullValue() && !isa<UndefValue>(Initializer)) {
          O << " = ";
          printScalarConstant(Initializer, O);
        }
      } else {
        // The frontend adds zero-initializer to device and constant variables
        // that don't have an initial value, and UndefValue to shared
        // variables, so skip warning for this case.
        if (!GVar->getInitializer()->isNullValue() &&
            !isa<UndefValue>(GVar->getInitializer())) {
          report_fatal_error("initial value of '" + GVar->getName() +
                             "' is not allowed in addrspace(" +
                             Twine(GVar->getAddressSpace()) + ")");
        }
      }
    }
  } else {
    // Although PTX has direct support for struct type and array type and
    // LLVM IR is very similar to PTX, the LLVM CodeGen does not support for
    // targets that support these high level field accesses. Structs, arrays
    // and vectors are lowered into arrays of bytes.
    switch (ETy->getTypeID()) {
    case Type::IntegerTyID: // Integers larger than 64 bits
    case Type::FP128TyID:
    case Type::StructTyID:
    case Type::ArrayTyID:
    case Type::FixedVectorTyID: {
      const uint64_t ElementSize = DL.getTypeStoreSize(ETy);
      // Ptx allows variable initilization only for constant and
      // global state spaces.
      if (((GVar->getAddressSpace() == ADDRESS_SPACE_GLOBAL) ||
           (GVar->getAddressSpace() == ADDRESS_SPACE_CONST)) &&
          GVar->hasInitializer()) {
        const Constant *Initializer = GVar->getInitializer();
        if (!isa<UndefValue>(Initializer) && !Initializer->isNullValue()) {
          AggBuffer aggBuffer(ElementSize, *this);
          bufferAggregateConstant(Initializer, &aggBuffer);
          if (aggBuffer.numSymbols()) {
            const unsigned int ptrSize = MAI->getCodePointerSize();
            if (ElementSize % ptrSize ||
                !aggBuffer.allSymbolsAligned(ptrSize)) {
              // Print in bytes and use the mask() operator for pointers.
              if (!STI.hasMaskOperator())
                report_fatal_error(
                    "initialized packed aggregate with pointers '" +
                    GVar->getName() +
                    "' requires at least PTX ISA version 7.1");
              O << " .u8 ";
              getSymbol(GVar)->print(O, MAI);
              O << "[" << ElementSize << "] = {";
              aggBuffer.printBytes(O);
              O << "}";
            } else {
              O << " .u" << ptrSize * 8 << " ";
              getSymbol(GVar)->print(O, MAI);
              O << "[" << ElementSize / ptrSize << "] = {";
              aggBuffer.printWords(O);
              O << "}";
            }
          } else {
            O << " .b8 ";
            getSymbol(GVar)->print(O, MAI);
            O << "[" << ElementSize << "] = {";
            aggBuffer.printBytes(O);
            O << "}";
          }
        } else {
          O << " .b8 ";
          getSymbol(GVar)->print(O, MAI);
          if (ElementSize)
            O << "[" << ElementSize << "]";
        }
      } else {
        O << " .b8 ";
        getSymbol(GVar)->print(O, MAI);
        if (ElementSize)
          O << "[" << ElementSize << "]";
      }
      break;
    }
    default:
      llvm_unreachable("type not supported yet");
    }
  }
  O << ";\n";
}

void NVPTXAsmPrinter::AggBuffer::printSymbol(unsigned nSym, raw_ostream &os) {
  const Value *v = Symbols[nSym];
  const Value *v0 = SymbolsBeforeStripping[nSym];
  if (const GlobalValue *GVar = dyn_cast<GlobalValue>(v)) {
    MCSymbol *Name = AP.getSymbol(GVar);
    PointerType *PTy = dyn_cast<PointerType>(v0->getType());
    // Is v0 a generic pointer?
    bool isGenericPointer = PTy && PTy->getAddressSpace() == 0;
    if (EmitGeneric && isGenericPointer && !isa<Function>(v)) {
      os << "generic(";
      Name->print(os, AP.MAI);
      os << ")";
    } else {
      Name->print(os, AP.MAI);
    }
  } else if (const ConstantExpr *CExpr = dyn_cast<ConstantExpr>(v0)) {
    const MCExpr *Expr = AP.lowerConstantForGV(CExpr, false);
    AP.printMCExpr(*Expr, os);
  } else
    llvm_unreachable("symbol type unknown");
}

void NVPTXAsmPrinter::AggBuffer::printBytes(raw_ostream &os) {
  unsigned int ptrSize = AP.MAI->getCodePointerSize();
  // Do not emit trailing zero initializers. They will be zero-initialized by
  // ptxas. This saves on both space requirements for the generated PTX and on
  // memory use by ptxas. (See:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#global-state-space)
  unsigned int InitializerCount = size;
  // TODO: symbols make this harder, but it would still be good to trim trailing
  // 0s for aggs with symbols as well.
  if (numSymbols() == 0)
    while (InitializerCount >= 1 && !buffer[InitializerCount - 1])
      InitializerCount--;

  symbolPosInBuffer.push_back(InitializerCount);
  unsigned int nSym = 0;
  unsigned int nextSymbolPos = symbolPosInBuffer[nSym];
  for (unsigned int pos = 0; pos < InitializerCount;) {
    if (pos)
      os << ", ";
    if (pos != nextSymbolPos) {
      os << (unsigned int)buffer[pos];
      ++pos;
      continue;
    }
    // Generate a per-byte mask() operator for the symbol, which looks like:
    //   .global .u8 addr[] = {0xFF(foo), 0xFF00(foo), 0xFF0000(foo), ...};
    // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#initializers
    std::string symText;
    llvm::raw_string_ostream oss(symText);
    printSymbol(nSym, oss);
    for (unsigned i = 0; i < ptrSize; ++i) {
      if (i)
        os << ", ";
      llvm::write_hex(os, 0xFFULL << i * 8, HexPrintStyle::PrefixUpper);
      os << "(" << symText << ")";
    }
    pos += ptrSize;
    nextSymbolPos = symbolPosInBuffer[++nSym];
    assert(nextSymbolPos >= pos);
  }
}

void NVPTXAsmPrinter::AggBuffer::printWords(raw_ostream &os) {
  unsigned int ptrSize = AP.MAI->getCodePointerSize();
  symbolPosInBuffer.push_back(size);
  unsigned int nSym = 0;
  unsigned int nextSymbolPos = symbolPosInBuffer[nSym];
  assert(nextSymbolPos % ptrSize == 0);
  for (unsigned int pos = 0; pos < size; pos += ptrSize) {
    if (pos)
      os << ", ";
    if (pos == nextSymbolPos) {
      printSymbol(nSym, os);
      nextSymbolPos = symbolPosInBuffer[++nSym];
      assert(nextSymbolPos % ptrSize == 0);
      assert(nextSymbolPos >= pos + ptrSize);
    } else if (ptrSize == 4)
      os << support::endian::read32le(&buffer[pos]);
    else
      os << support::endian::read64le(&buffer[pos]);
  }
}

void NVPTXAsmPrinter::emitDemotedVars(const Function *F, raw_ostream &O) {
  auto It = localDecls.find(F);
  if (It == localDecls.end())
    return;

  ArrayRef<const GlobalVariable *> GVars = It->second;

  const NVPTXTargetMachine &NTM = static_cast<const NVPTXTargetMachine &>(TM);
  const NVPTXSubtarget &STI =
      *static_cast<const NVPTXSubtarget *>(NTM.getSubtargetImpl());

  for (const GlobalVariable *GV : GVars) {
    O << "\t// demoted variable\n\t";
    printModuleLevelGV(GV, O, /*processDemoted=*/true, STI);
  }
}

void NVPTXAsmPrinter::emitPTXAddressSpace(unsigned int AddressSpace,
                                          raw_ostream &O) const {
  switch (AddressSpace) {
  case ADDRESS_SPACE_LOCAL:
    O << "local";
    break;
  case ADDRESS_SPACE_GLOBAL:
    O << "global";
    break;
  case ADDRESS_SPACE_CONST:
    O << "const";
    break;
  case ADDRESS_SPACE_SHARED:
    O << "shared";
    break;
  default:
    report_fatal_error("Bad address space found while emitting PTX: " +
                       llvm::Twine(AddressSpace));
    break;
  }
}

std::string
NVPTXAsmPrinter::getPTXFundamentalTypeStr(Type *Ty, bool useB4PTR) const {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1)
      return "pred";
    if (NumBits <= 64) {
      std::string name = "u";
      return name + utostr(NumBits);
    }
    llvm_unreachable("Integer too large");
    break;
  }
  case Type::BFloatTyID:
  case Type::HalfTyID:
    // fp16 and bf16 are stored as .b16 for compatibility with pre-sm_53
    // PTX assembly.
    return "b16";
  case Type::FloatTyID:
    return "f32";
  case Type::DoubleTyID:
    return "f64";
  case Type::PointerTyID: {
    unsigned PtrSize = TM.getPointerSizeInBits(Ty->getPointerAddressSpace());
    assert((PtrSize == 64 || PtrSize == 32) && "Unexpected pointer size");

    if (PtrSize == 64)
      if (useB4PTR)
        return "b64";
      else
        return "u64";
    else if (useB4PTR)
      return "b32";
    else
      return "u32";
  }
  default:
    break;
  }
  llvm_unreachable("unexpected type");
}

void NVPTXAsmPrinter::emitPTXGlobalVariable(const GlobalVariable *GVar,
                                            raw_ostream &O,
                                            const NVPTXSubtarget &STI) {
  const DataLayout &DL = getDataLayout();

  // GlobalVariables are always constant pointers themselves.
  Type *ETy = GVar->getValueType();

  O << ".";
  emitPTXAddressSpace(GVar->getType()->getAddressSpace(), O);
  if (isManaged(*GVar)) {
    if (STI.getPTXVersion() < 40 || STI.getSmVersion() < 30)
      report_fatal_error(
          ".attribute(.managed) requires PTX version >= 4.0 and sm_30");

    O << " .attribute(.managed)";
  }
  O << " .align "
    << GVar->getAlign().value_or(DL.getPrefTypeAlign(ETy)).value();

  // Special case for i128/fp128
  if (ETy->getScalarSizeInBits() == 128) {
    O << " .b8 ";
    getSymbol(GVar)->print(O, MAI);
    O << "[16]";
    return;
  }

  if (ETy->isFloatingPointTy() || ETy->isIntOrPtrTy()) {
    O << " ." << getPTXFundamentalTypeStr(ETy) << " ";
    getSymbol(GVar)->print(O, MAI);
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
  case Type::FixedVectorTyID:
    ElementSize = DL.getTypeStoreSize(ETy);
    O << " .b8 ";
    getSymbol(GVar)->print(O, MAI);
    O << "[";
    if (ElementSize) {
      O << ElementSize;
    }
    O << "]";
    break;
  default:
    llvm_unreachable("type not supported yet");
  }
}

void NVPTXAsmPrinter::emitFunctionParamList(const Function *F, raw_ostream &O) {
  const DataLayout &DL = getDataLayout();
  const NVPTXSubtarget &STI = TM.getSubtarget<NVPTXSubtarget>(*F);
  const auto *TLI = cast<NVPTXTargetLowering>(STI.getTargetLowering());
  const NVPTXMachineFunctionInfo *MFI =
      MF ? MF->getInfo<NVPTXMachineFunctionInfo>() : nullptr;

  bool IsFirst = true;
  const bool IsKernelFunc = isKernelFunction(*F);

  if (F->arg_empty() && !F->isVarArg()) {
    O << "()";
    return;
  }

  O << "(\n";

  for (const Argument &Arg : F->args()) {
    Type *Ty = Arg.getType();
    const std::string ParamSym = TLI->getParamName(F, Arg.getArgNo());

    if (!IsFirst)
      O << ",\n";

    IsFirst = false;

    // Handle image/sampler parameters
    if (IsKernelFunc) {
      const bool IsSampler = isSampler(Arg);
      const bool IsTexture = !IsSampler && isImageReadOnly(Arg);
      const bool IsSurface = !IsSampler && !IsTexture &&
                             (isImageReadWrite(Arg) || isImageWriteOnly(Arg));
      if (IsSampler || IsTexture || IsSurface) {
        const bool EmitImgPtr = !MFI || !MFI->checkImageHandleSymbol(ParamSym);
        O << "\t.param ";
        if (EmitImgPtr)
          O << ".u64 .ptr ";

        if (IsSampler)
          O << ".samplerref ";
        else if (IsTexture)
          O << ".texref ";
        else // IsSurface
          O << ".surfref ";
        O << ParamSym;
        continue;
      }
    }

    auto GetOptimalAlignForParam = [TLI, &DL, F, &Arg](Type *Ty) -> Align {
      if (MaybeAlign StackAlign =
              getAlign(*F, Arg.getArgNo() + AttributeList::FirstArgIndex))
        return StackAlign.value();

      Align TypeAlign = TLI->getFunctionParamOptimizedAlign(F, Ty, DL);
      MaybeAlign ParamAlign =
          Arg.hasByValAttr() ? Arg.getParamAlign() : MaybeAlign();
      return std::max(TypeAlign, ParamAlign.valueOrOne());
    };

    if (Arg.hasByValAttr()) {
      // param has byVal attribute.
      Type *ETy = Arg.getParamByValType();
      assert(ETy && "Param should have byval type");

      // Print .param .align <a> .b8 .param[size];
      // <a>  = optimal alignment for the element type; always multiple of
      //        PAL.getParamAlignment
      // size = typeallocsize of element type
      const Align OptimalAlign =
          IsKernelFunc ? GetOptimalAlignForParam(ETy)
                       : TLI->getFunctionByValParamAlign(
                             F, ETy, Arg.getParamAlign().valueOrOne(), DL);

      O << "\t.param .align " << OptimalAlign.value() << " .b8 " << ParamSym
        << "[" << DL.getTypeAllocSize(ETy) << "]";
      continue;
    }

    if (shouldPassAsArray(Ty)) {
      // Just print .param .align <a> .b8 .param[size];
      // <a>  = optimal alignment for the element type; always multiple of
      //        PAL.getParamAlignment
      // size = typeallocsize of element type
      Align OptimalAlign = GetOptimalAlignForParam(Ty);

      O << "\t.param .align " << OptimalAlign.value() << " .b8 " << ParamSym
        << "[" << DL.getTypeAllocSize(Ty) << "]";

      continue;
    }
    // Just a scalar
    auto *PTy = dyn_cast<PointerType>(Ty);
    unsigned PTySizeInBits = 0;
    if (PTy) {
      PTySizeInBits =
          TLI->getPointerTy(DL, PTy->getAddressSpace()).getSizeInBits();
      assert(PTySizeInBits && "Invalid pointer size");
    }

    if (IsKernelFunc) {
      if (PTy) {
        O << "\t.param .u" << PTySizeInBits << " .ptr";

        switch (PTy->getAddressSpace()) {
        default:
          break;
        case ADDRESS_SPACE_GLOBAL:
          O << " .global";
          break;
        case ADDRESS_SPACE_SHARED:
          O << " .shared";
          break;
        case ADDRESS_SPACE_CONST:
          O << " .const";
          break;
        case ADDRESS_SPACE_LOCAL:
          O << " .local";
          break;
        }

        O << " .align " << Arg.getParamAlign().valueOrOne().value() << " "
          << ParamSym;
        continue;
      }

      // non-pointer scalar to kernel func
      O << "\t.param .";
      // Special case: predicate operands become .u8 types
      if (Ty->isIntegerTy(1))
        O << "u8";
      else
        O << getPTXFundamentalTypeStr(Ty);
      O << " " << ParamSym;
      continue;
    }
    // Non-kernel function, just print .param .b<size> for ABI
    // and .reg .b<size> for non-ABI
    unsigned Size;
    if (auto *ITy = dyn_cast<IntegerType>(Ty)) {
      Size = promoteScalarArgumentSize(ITy->getBitWidth());
    } else if (PTy) {
      assert(PTySizeInBits && "Invalid pointer size");
      Size = PTySizeInBits;
    } else
      Size = Ty->getPrimitiveSizeInBits();
    O << "\t.param .b" << Size << " " << ParamSym;
  }

  if (F->isVarArg()) {
    if (!IsFirst)
      O << ",\n";
    O << "\t.param .align " << STI.getMaxRequiredAlignment() << " .b8 "
      << TLI->getParamName(F, /* vararg */ -1) << "[]";
  }

  O << "\n)";
}

void NVPTXAsmPrinter::setAndEmitFunctionVirtualRegisters(
    const MachineFunction &MF) {
  SmallString<128> Str;
  raw_svector_ostream O(Str);

  // Map the global virtual register number to a register class specific
  // virtual register number starting from 1 with that class.
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  //unsigned numRegClasses = TRI->getNumRegClasses();

  // Emit the Fake Stack Object
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  int64_t NumBytes = MFI.getStackSize();
  if (NumBytes) {
    O << "\t.local .align " << MFI.getMaxAlign().value() << " .b8 \t"
      << DEPOTNAME << getFunctionNumber() << "[" << NumBytes << "];\n";
    if (static_cast<const NVPTXTargetMachine &>(MF.getTarget()).is64Bit()) {
      O << "\t.reg .b64 \t%SP;\n"
        << "\t.reg .b64 \t%SPL;\n";
    } else {
      O << "\t.reg .b32 \t%SP;\n"
        << "\t.reg .b32 \t%SPL;\n";
    }
  }

  // Go through all virtual registers to establish the mapping between the
  // global virtual
  // register number and the per class virtual register number.
  // We use the per class virtual register number in the ptx output.
  unsigned int numVRs = MRI->getNumVirtRegs();
  for (unsigned i = 0; i < numVRs; i++) {
    Register vr = Register::index2VirtReg(i);
    const TargetRegisterClass *RC = MRI->getRegClass(vr);
    DenseMap<unsigned, unsigned> &regmap = VRegMapping[RC];
    int n = regmap.size();
    regmap.insert(std::make_pair(vr, n + 1));
  }

  // Emit declaration of the virtual registers or 'physical' registers for
  // each register class
  for (const TargetRegisterClass *RC : TRI->regclasses()) {
    const unsigned N = VRegMapping[RC].size();

    // Only declare those registers that may be used.
    if (N) {
      const StringRef RCName = getNVPTXRegClassName(RC);
      const StringRef RCStr = getNVPTXRegClassStr(RC);
      O << "\t.reg " << RCName << " \t" << RCStr << "<" << (N + 1) << ">;\n";
    }
  }

  OutStreamer->emitRawText(O.str());
}

/// Translate virtual register numbers in DebugInfo locations to their printed
/// encodings, as used by CUDA-GDB.
void NVPTXAsmPrinter::encodeDebugInfoRegisterNumbers(
    const MachineFunction &MF) {
  const NVPTXSubtarget &STI = MF.getSubtarget<NVPTXSubtarget>();
  const NVPTXRegisterInfo *registerInfo = STI.getRegisterInfo();

  // Clear the old mapping, and add the new one.  This mapping is used after the
  // printing of the current function is complete, but before the next function
  // is printed.
  registerInfo->clearDebugRegisterMap();

  for (auto &classMap : VRegMapping) {
    for (auto &registerMapping : classMap.getSecond()) {
      auto reg = registerMapping.getFirst();
      registerInfo->addToDebugRegisterMap(reg, getVirtualRegisterName(reg));
    }
  }
}

void NVPTXAsmPrinter::printFPConstant(const ConstantFP *Fp,
                                      raw_ostream &O) const {
  APFloat APF = APFloat(Fp->getValueAPF()); // make a copy
  bool ignored;
  unsigned int numHex;
  const char *lead;

  if (Fp->getType()->getTypeID() == Type::FloatTyID) {
    numHex = 8;
    lead = "0f";
    APF.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &ignored);
  } else if (Fp->getType()->getTypeID() == Type::DoubleTyID) {
    numHex = 16;
    lead = "0d";
    APF.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &ignored);
  } else
    llvm_unreachable("unsupported fp type");

  APInt API = APF.bitcastToAPInt();
  O << lead << format_hex_no_prefix(API.getZExtValue(), numHex, /*Upper=*/true);
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
    const bool IsNonGenericPointer = GVar->getAddressSpace() != 0;
    if (EmitGeneric && !isa<Function>(CPV) && !IsNonGenericPointer) {
      O << "generic(";
      getSymbol(GVar)->print(O, MAI);
      O << ")";
    } else {
      getSymbol(GVar)->print(O, MAI);
    }
    return;
  }
  if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
    const MCExpr *E = lowerConstantForGV(cast<Constant>(Cexpr), false);
    printMCExpr(*E, O);
    return;
  }
  llvm_unreachable("Not scalar type found in printScalarConstant()");
}

void NVPTXAsmPrinter::bufferLEByte(const Constant *CPV, int Bytes,
                                   AggBuffer *AggBuffer) {
  const DataLayout &DL = getDataLayout();
  int AllocSize = DL.getTypeAllocSize(CPV->getType());
  if (isa<UndefValue>(CPV) || CPV->isNullValue()) {
    // Non-zero Bytes indicates that we need to zero-fill everything. Otherwise,
    // only the space allocated by CPV.
    AggBuffer->addZeros(Bytes ? Bytes : AllocSize);
    return;
  }

  // Helper for filling AggBuffer with APInts.
  auto AddIntToBuffer = [AggBuffer, Bytes](const APInt &Val) {
    size_t NumBytes = (Val.getBitWidth() + 7) / 8;
    SmallVector<unsigned char, 16> Buf(NumBytes);
    // `extractBitsAsZExtValue` does not allow the extraction of bits beyond the
    // input's bit width, and i1 arrays may not have a length that is a multuple
    // of 8. We handle the last byte separately, so we never request out of
    // bounds bits.
    for (unsigned I = 0; I < NumBytes - 1; ++I) {
      Buf[I] = Val.extractBitsAsZExtValue(8, I * 8);
    }
    size_t LastBytePosition = (NumBytes - 1) * 8;
    size_t LastByteBits = Val.getBitWidth() - LastBytePosition;
    Buf[NumBytes - 1] =
        Val.extractBitsAsZExtValue(LastByteBits, LastBytePosition);
    AggBuffer->addBytes(Buf.data(), NumBytes, Bytes);
  };

  switch (CPV->getType()->getTypeID()) {
  case Type::IntegerTyID:
    if (const auto *CI = dyn_cast<ConstantInt>(CPV)) {
      AddIntToBuffer(CI->getValue());
      break;
    }
    if (const auto *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
      if (const auto *CI =
              dyn_cast<ConstantInt>(ConstantFoldConstant(Cexpr, DL))) {
        AddIntToBuffer(CI->getValue());
        break;
      }
      if (Cexpr->getOpcode() == Instruction::PtrToInt) {
        Value *V = Cexpr->getOperand(0)->stripPointerCasts();
        AggBuffer->addSymbol(V, Cexpr->getOperand(0));
        AggBuffer->addZeros(AllocSize);
        break;
      }
    }
    llvm_unreachable("unsupported integer const type");
    break;

  case Type::HalfTyID:
  case Type::BFloatTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
    AddIntToBuffer(cast<ConstantFP>(CPV)->getValueAPF().bitcastToAPInt());
    break;

  case Type::PointerTyID: {
    if (const GlobalValue *GVar = dyn_cast<GlobalValue>(CPV)) {
      AggBuffer->addSymbol(GVar, GVar);
    } else if (const ConstantExpr *Cexpr = dyn_cast<ConstantExpr>(CPV)) {
      const Value *v = Cexpr->stripPointerCasts();
      AggBuffer->addSymbol(v, Cexpr);
    }
    AggBuffer->addZeros(AllocSize);
    break;
  }

  case Type::ArrayTyID:
  case Type::FixedVectorTyID:
  case Type::StructTyID: {
    if (isa<ConstantAggregate>(CPV) || isa<ConstantDataSequential>(CPV)) {
      bufferAggregateConstant(CPV, AggBuffer);
      if (Bytes > AllocSize)
        AggBuffer->addZeros(Bytes - AllocSize);
    } else if (isa<ConstantAggregateZero>(CPV))
      AggBuffer->addZeros(Bytes);
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
  const DataLayout &DL = getDataLayout();

  auto ExtendBuffer = [](APInt Val, AggBuffer *Buffer) {
    for (unsigned I : llvm::seq(Val.getBitWidth() / 8))
      Buffer->addByte(Val.extractBitsAsZExtValue(8, I * 8));
  };

  // Integers of arbitrary width
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CPV)) {
    ExtendBuffer(CI->getValue(), aggBuffer);
    return;
  }

  // f128
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CPV)) {
    if (CFP->getType()->isFP128Ty()) {
      ExtendBuffer(CFP->getValueAPF().bitcastToAPInt(), aggBuffer);
      return;
    }
  }

  // Old constants
  if (isa<ConstantArray>(CPV) || isa<ConstantVector>(CPV)) {
    for (const auto &Op : CPV->operands())
      bufferLEByte(cast<Constant>(Op), 0, aggBuffer);
    return;
  }

  if (const auto *CDS = dyn_cast<ConstantDataSequential>(CPV)) {
    for (unsigned I : llvm::seq(CDS->getNumElements()))
      bufferLEByte(cast<Constant>(CDS->getElementAsConstant(I)), 0, aggBuffer);
    return;
  }

  if (isa<ConstantStruct>(CPV)) {
    if (CPV->getNumOperands()) {
      StructType *ST = cast<StructType>(CPV->getType());
      for (unsigned I : llvm::seq(CPV->getNumOperands())) {
        int EndOffset = (I + 1 == CPV->getNumOperands())
                            ? DL.getStructLayout(ST)->getElementOffset(0) +
                                  DL.getTypeAllocSize(ST)
                            : DL.getStructLayout(ST)->getElementOffset(I + 1);
        int Bytes = EndOffset - DL.getStructLayout(ST)->getElementOffset(I);
        bufferLEByte(cast<Constant>(CPV->getOperand(I)), Bytes, aggBuffer);
      }
    }
    return;
  }
  llvm_unreachable("unsupported constant type in printAggregateConstant()");
}

/// lowerConstantForGV - Return an MCExpr for the given Constant.  This is mostly
/// a copy from AsmPrinter::lowerConstant, except customized to only handle
/// expressions that are representable in PTX and create
/// NVPTXGenericMCSymbolRefExpr nodes for addrspacecast instructions.
const MCExpr *
NVPTXAsmPrinter::lowerConstantForGV(const Constant *CV,
                                    bool ProcessingGeneric) const {
  MCContext &Ctx = OutContext;

  if (CV->isNullValue() || isa<UndefValue>(CV))
    return MCConstantExpr::create(0, Ctx);

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV))
    return MCConstantExpr::create(CI->getZExtValue(), Ctx);

  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(getSymbol(GV), Ctx);
    if (ProcessingGeneric)
      return NVPTXGenericMCSymbolRefExpr::create(Expr, Ctx);
    return Expr;
  }

  const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV);
  if (!CE) {
    llvm_unreachable("Unknown constant value to lower!");
  }

  switch (CE->getOpcode()) {
  default:
    break; // Error

  case Instruction::AddrSpaceCast: {
    // Strip the addrspacecast and pass along the operand
    PointerType *DstTy = cast<PointerType>(CE->getType());
    if (DstTy->getAddressSpace() == 0)
      return lowerConstantForGV(cast<const Constant>(CE->getOperand(0)), true);

    break; // Error
  }

  case Instruction::GetElementPtr: {
    const DataLayout &DL = getDataLayout();

    // Generate a symbolic expression for the byte address
    APInt OffsetAI(DL.getPointerTypeSizeInBits(CE->getType()), 0);
    cast<GEPOperator>(CE)->accumulateConstantOffset(DL, OffsetAI);

    const MCExpr *Base = lowerConstantForGV(CE->getOperand(0),
                                            ProcessingGeneric);
    if (!OffsetAI)
      return Base;

    int64_t Offset = OffsetAI.getSExtValue();
    return MCBinaryExpr::createAdd(Base, MCConstantExpr::create(Offset, Ctx),
                                   Ctx);
  }

  case Instruction::Trunc:
    // We emit the value and depend on the assembler to truncate the generated
    // expression properly.  This is important for differences between
    // blockaddress labels.  Since the two labels are in the same function, it
    // is reasonable to treat their delta as a 32-bit value.
    [[fallthrough]];
  case Instruction::BitCast:
    return lowerConstantForGV(CE->getOperand(0), ProcessingGeneric);

  case Instruction::IntToPtr: {
    const DataLayout &DL = getDataLayout();

    // Handle casts to pointers by changing them into casts to the appropriate
    // integer type.  This promotes constant folding and simplifies this code.
    Constant *Op = CE->getOperand(0);
    Op = ConstantFoldIntegerCast(Op, DL.getIntPtrType(CV->getType()),
                                 /*IsSigned*/ false, DL);
    if (Op)
      return lowerConstantForGV(Op, ProcessingGeneric);

    break; // Error
  }

  case Instruction::PtrToInt: {
    const DataLayout &DL = getDataLayout();

    // Support only foldable casts to/from pointers that can be eliminated by
    // changing the pointer to the appropriately sized integer type.
    Constant *Op = CE->getOperand(0);
    Type *Ty = CE->getType();

    const MCExpr *OpExpr = lowerConstantForGV(Op, ProcessingGeneric);

    // We can emit the pointer value into this slot if the slot is an
    // integer slot equal to the size of the pointer.
    if (DL.getTypeAllocSize(Ty) == DL.getTypeAllocSize(Op->getType()))
      return OpExpr;

    // Otherwise the pointer is smaller than the resultant integer, mask off
    // the high bits so we are sure to get a proper truncation if the input is
    // a constant expr.
    unsigned InBits = DL.getTypeAllocSizeInBits(Op->getType());
    const MCExpr *MaskExpr = MCConstantExpr::create(~0ULL >> (64-InBits), Ctx);
    return MCBinaryExpr::createAnd(OpExpr, MaskExpr, Ctx);
  }

  // The MC library also has a right-shift operator, but it isn't consistently
  // signed or unsigned between different targets.
  case Instruction::Add: {
    const MCExpr *LHS = lowerConstantForGV(CE->getOperand(0), ProcessingGeneric);
    const MCExpr *RHS = lowerConstantForGV(CE->getOperand(1), ProcessingGeneric);
    switch (CE->getOpcode()) {
    default: llvm_unreachable("Unknown binary operator constant cast expr");
    case Instruction::Add: return MCBinaryExpr::createAdd(LHS, RHS, Ctx);
    }
  }
  }

  // If the code isn't optimized, there may be outstanding folding
  // opportunities. Attempt to fold the expression using DataLayout as a
  // last resort before giving up.
  Constant *C = ConstantFoldConstant(CE, getDataLayout());
  if (C != CE)
    return lowerConstantForGV(C, ProcessingGeneric);

  // Otherwise report the problem to the user.
  std::string S;
  raw_string_ostream OS(S);
  OS << "Unsupported expression in static initializer: ";
  CE->printAsOperand(OS, /*PrintType=*/false,
                 !MF ? nullptr : MF->getFunction().getParent());
  report_fatal_error(Twine(OS.str()));
}

void NVPTXAsmPrinter::printMCExpr(const MCExpr &Expr, raw_ostream &OS) const {
  OutContext.getAsmInfo()->printExpr(OS, Expr);
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool NVPTXAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O);
    case 'r':
      break;
    }
  }

  printOperand(MI, OpNo, O);

  return false;
}

bool NVPTXAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier

  O << '[';
  printMemOperand(MI, OpNo, O);
  O << ']';

  return false;
}

void NVPTXAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (MO.getReg().isPhysical()) {
      if (MO.getReg() == NVPTX::VRDepot)
        O << DEPOTNAME << getFunctionNumber();
      else
        O << NVPTXInstPrinter::getRegisterName(MO.getReg());
    } else {
      emitVirtualRegister(MO.getReg(), O);
    }
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_FPImmediate:
    printFPConstant(MO.getFPImm(), O);
    break;

  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, O);
    break;

  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    break;

  default:
    llvm_unreachable("Operand type not supported.");
  }
}

void NVPTXAsmPrinter::printMemOperand(const MachineInstr *MI, unsigned OpNum,
                                      raw_ostream &O, const char *Modifier) {
  printOperand(MI, OpNum, O);

  if (Modifier && strcmp(Modifier, "add") == 0) {
    O << ", ";
    printOperand(MI, OpNum + 1, O);
  } else {
    if (MI->getOperand(OpNum + 1).isImm() &&
        MI->getOperand(OpNum + 1).getImm() == 0)
      return; // don't print ',0' or '+0'
    O << "+";
    printOperand(MI, OpNum + 1, O);
  }
}

char NVPTXAsmPrinter::ID = 0;

INITIALIZE_PASS(NVPTXAsmPrinter, "nvptx-asm-printer", "NVPTX Assembly Printer",
                false, false)

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNVPTXAsmPrinter() {
  RegisterAsmPrinter<NVPTXAsmPrinter> X(getTheNVPTXTarget32());
  RegisterAsmPrinter<NVPTXAsmPrinter> Y(getTheNVPTXTarget64());
}
