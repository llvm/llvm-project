//===-- PISAAsmPrinter.cpp - PISA LLVM assembly writer --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAInstPrinter.h"
#include "MCTargetDesc/PISARegEncoder.h"
#include "MCTargetDesc/PISATargetStreamer.h"
#include "PISA.h"
#include "PISAInstrInfo.h"
#include "PISAMCInstLower.h"
#include "PISAMachineFunctionInfo.h"
#include "PISARegManager.h"
#include "PISASubtarget.h"
#include "PISATargetMachine.h"
#include "PISAUtils.h"
#include "TargetInfo/PISATargetInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PISAAddrSpace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/PISATargetParser.h"
#include <llvm/IR/DiagnosticInfo.h>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class PISAAsmPrinter : public AsmPrinter {

public:
  PISATargetStreamer &getTargetStreamer() const {
    return static_cast<PISATargetStreamer &>(*OutStreamer->getTargetStreamer());
  }

private:
  void collectRegDcls(PISA::RegDcls &);
  void collectLocalVariableDcls(PISA::LocalVariableDcls &);
  void updateFuncParamIdxs(PISA::DataTypes &DTs);

  void outputInstruction(const MachineInstr *MI);
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);

  std::string getVirtualRegisterName(Register R) const;

  void collectFunctionDeclaration(PISA::FunctionDeclaration &,
                                  const Function &F);
  void collectFunctionSignature(PISA::FunctionSignature &);
  void collectFunctionParameters(PISA::FunctionSignature &);
  void collectKernelParameters(PISA::FunctionSignature &);
  void collectFunctionDirectiveAndName(PISA::FunctionDirectiveAndName &DN,
                                       const Function &F);
  PISA::LinkageTy collectLinkage(const GlobalValue &V);
  void collectGlobalVariable(PISA::GlobalVariableDcl &PGV,
                             const GlobalVariable &GV);

  void emitGlobalsAndFuncDecls(Module &M);

  const PISASubtarget *ST = nullptr;
  const PISAInstrInfo *TII = nullptr;
  const PISARegisterInfo *TRI = nullptr;
  PISA::RegManager *RegMgr = nullptr;
  PISA::DataTypes *DTs = nullptr;
  bool GlobalsEmitted = false;

  class FlattenGlobal {
  public:
    FlattenGlobal(const Constant *C, PISA::VariableInit &VI,
                  const DataLayout &DL, AsmPrinter &AP)
        : DL(DL), VI(VI), AP(AP) {
      process(C);
      dischargeZeros();
      assert(computeSize(C) == DL.getTypeAllocSize(C->getType()) &&
             "size mismatch?");
    }

  private:
    bool isZero(const Constant *C) const {
      if (isa<ConstantPointerNull>(C))
        return false;

      return C->isNullValue() || isa<UndefValue>(C);
    }
    void pad(const Constant *C, unsigned NumElts = 0) {
      unsigned Size = DL.getTypeAllocSize(C->getType());
      if (NumElts == 0) {
        ZeroCnt += Size;
        return;
      }
      unsigned EmittedSize =
          DL.getTypeAllocSize(C->getType()->getContainedType(0)) * NumElts;
      assert(EmittedSize <= Size && "Size cannot be less than EmittedSize!");
      if (unsigned Padding = Size - EmittedSize)
        ZeroCnt += Padding;
    }
    void pad(uint64_t NumBytes) { ZeroCnt += NumBytes; }
    void dischargeZeros() {
      if (ZeroCnt == 0)
        return;
      // Insert dummy slot
      VI.Initializer.push_back({LLT{}, 0});
      uint64_t Idx = VI.Initializer.size() - 1;
      VI.Exprs.insert({Idx, PISA::VariableInit::Zeros{ZeroCnt}});
      ZeroCnt = 0;
    }
    void addVal(LLT Ty, uint64_t Val) {
      dischargeZeros();
      VI.Initializer.push_back({Ty, Val});
    }
    void addGlobal(LLT Ty, const PISA::VariableInit::GlobalExpr &GE) {
      // Insert dummy slot
      addVal(Ty, 0);
      uint64_t Idx = VI.Initializer.size() - 1;
      VI.Exprs.insert({Idx, GE});
    }
    void lowerConstant(const Constant *C) {
      auto *Expr = AP.lowerConstant(C);
      MCValue Res;
      if (!Expr->evaluateAsRelocatable(Res, nullptr))
        llvm_unreachable("unhandled expression!");
      LLT Ty = getLLTForType(*C->getType(), DL);
      if (!Res.getAddSym() && !Res.getSubSym()) {
        if (Res.getConstant() == 0)
          pad(C);
        else
          addVal(Ty, static_cast<uint64_t>(Res.getConstant()));
        return;
      }
      assert(!Res.getSubSym() && "unhandled expression!");
      std::string Name = Res.getAddSym()->getName().str();
      PISA::VariableInit::GlobalExpr E{std::move(Name), Res.getConstant()};
      addGlobal(Ty, E);
    }
    uint64_t computeSize(const Constant *C) const {
      uint64_t Total = 0;
      for (auto [i, Elt] : llvm::enumerate(VI.Initializer)) {
        if (auto Iter = VI.Exprs.find(i); Iter != VI.Exprs.end()) {
          auto &Entry = Iter->second;
          if (auto *Z = std::get_if<PISA::VariableInit::Zeros>(&Entry)) {
            Total += Z->N;
            continue;
          }
        }
        Total += Elt.Type.getSizeInBytes();
      }
      return Total;
    }
    void emitGlobalConstantLargeInt(const ConstantInt *CI) {
      unsigned BitWidth = CI->getBitWidth();

      // Copy the value as we may massage the layout for constants whose bit
      // width is not a multiple of 64-bits.
      APInt Realigned(CI->getValue());
      uint64_t ExtraBits = 0;
      unsigned ExtraBitsSize = BitWidth & 63;

      if (ExtraBitsSize) {
        // The bit width of the data is not a multiple of 64-bits.
        // The extra bits are expected to be at the end of the chunk of the
        // memory. Little endian:
        // * Nothing to be done, just record the extra bits to emit.
        ExtraBits = Realigned.getRawData()[BitWidth / 64];
      }

      // We don't expect assemblers to support integer data directives
      // for more than 64 bits, so we emit the data in at most 64-bit
      // quantities at a time.
      const uint64_t *RawData = Realigned.getRawData();
      for (unsigned I = 0, E = BitWidth / 64; I != E; ++I)
        addVal(LLT::integer(64), RawData[I]);

      if (ExtraBitsSize) {
        // Emit the extra bits after the 64-bits chunks.
        // Emit a directive that fills the expected size.
        uint64_t Size = DL.getTypeStoreSize(CI->getType());
        Size -= (BitWidth / 64) * 8;
        assert(Size && Size * 8 >= ExtraBitsSize &&
               (ExtraBits & (((uint64_t)-1) >> (64 - ExtraBitsSize))) ==
                   ExtraBits &&
               "Directive too small for extra bits.");
        addVal(LLT::integer(Size * 8), ExtraBits);
      }
    }
    const DataLayout &DL;
    PISA::VariableInit &VI;
    AsmPrinter &AP;
    void process(const Constant *C);
    unsigned ZeroCnt = 0;
  };

protected:
  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;

public:
  explicit PISAAsmPrinter(TargetMachine &TM,
                          std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "PISA Assembly Printer"; }
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;

  void emitInstruction(const MachineInstr *MI) override;
  void emitFunctionHeader() override;
  void emitFunctionBodyStart() override;
  void emitFunctionBodyEnd() override;
  void emitEndOfAsmFile(Module &) override;

  void emitFunctionEntryLabel() override {}
  void emitBasicBlockEnd(const MachineBasicBlock &MBB) override {}
  void emitGlobalVariable(const GlobalVariable *GV) override {}

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

void PISAAsmPrinter::FlattenGlobal::process(const Constant *C) {
  uint64_t Size = DL.getTypeAllocSize(C->getType());
  if (isZero(C))
    return pad(C);
  auto AddSplatVector = [&](LLT ScalarTy, const APInt &EltVal) {
    assert(EltVal.getBitWidth() <= 64 && "Splat element too wide for uint64_t");
    auto *VTy = cast<FixedVectorType>(C->getType());
    unsigned NumElts = VTy->getNumElements();
    uint64_t Val = EltVal.getZExtValue();
    for (unsigned I = 0; I < NumElts; ++I)
      addVal(ScalarTy, Val);
    pad(C, NumElts);
  };

  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    if (C->getType()->isVectorTy()) {
      auto *VTy = cast<FixedVectorType>(C->getType());
      uint64_t EltAllocSize = DL.getTypeAllocSize(VTy->getElementType());
      LLT ScalarTy = LLT::integer(EltAllocSize * 8);
      AddSplatVector(ScalarTy, CI->getValue());
    } else {
      // We don't use the LLT type of `C` directly here because `C` could be,
      // for example, a s1. The allocation size is 1, so we want to give it
      // a type of s8 to reflect that.
      addVal(LLT::integer(Size * 8), CI->getZExtValue());
    }
  } else if (auto *FP = dyn_cast<ConstantFP>(C)) {
    if (C->getType()->isVectorTy()) {
      auto *VTy = cast<FixedVectorType>(C->getType());
      LLT ScalarTy = getLLTForType(*VTy->getElementType(), DL);
      AddSplatVector(ScalarTy, FP->getValueAPF().bitcastToAPInt());
    } else {
      LLT Ty = getLLTForType(*C->getType(), DL);
      addVal(Ty, FP->getValueAPF().bitcastToAPInt().getZExtValue());
    }
  } else if (isa<ConstantPointerNull>(C)) {
    LLT Ty = getLLTForType(*C->getType(), DL);
    unsigned AS = C->getType()->getPointerAddressSpace();
    if (uint64_t Val = PISATargetMachine::getNullPointerValue(AS))
      addVal(Ty, Val);
    else
      pad(C);
  } else if (auto *CV = dyn_cast<ConstantVector>(C)) {
    Type *ElementType = CV->getType()->getElementType();
    uint64_t ElementSizeInBits = DL.getTypeSizeInBits(ElementType);
    uint64_t ElementAllocSizeInBits = DL.getTypeAllocSizeInBits(ElementType);
    if (ElementSizeInBits != ElementAllocSizeInBits) {
      // If the allocation size of an element is different from the size in
      // bits, printing each element separately will insert incorrect padding.
      //
      // The general algorithm here is complicated; instead of writing it out
      // here, just use the existing code in ConstantFolding.
      Type *IntT = IntegerType::get(CV->getContext(),
                                    DL.getTypeSizeInBits(CV->getType()));
      ConstantInt *CI = dyn_cast_or_null<ConstantInt>(ConstantFoldConstant(
          ConstantExpr::getBitCast(const_cast<ConstantVector *>(CV), IntT),
          DL));
      if (!CI) {
        report_fatal_error(
            "Cannot lower vector global with unusual element type");
      }
      emitGlobalConstantLargeInt(CI);
      uint64_t EmittedSize = DL.getTypeStoreSize(CV->getType());
      if (unsigned Padding = Size - EmittedSize)
        pad(Padding);
    } else {
      for (unsigned I = 0; I < CV->getNumOperands(); I++)
        process(CV->getAggregateElement(I));
      pad(C, CV->getNumOperands());
    }
  } else if (auto *CA = dyn_cast<ConstantArray>(C)) {
    for (unsigned I = 0; I < CA->getNumOperands(); I++)
      process(CA->getAggregateElement(I));
  } else if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    auto *StructTy = cast<StructType>(CS->getType());
    auto *Layout = DL.getStructLayout(StructTy);
    for (unsigned I = 0, E = CS->getNumOperands(); I != E; ++I) {
      const Constant *Field = CS->getOperand(I);
      // Print the actual field value.
      process(Field);
      // Check if padding is needed and insert one or more 0s.
      uint64_t FieldSize = DL.getTypeAllocSize(Field->getType());
      uint64_t PadSize =
          ((I == E - 1 ? Size : Layout->getElementOffset(I + 1)) -
           Layout->getElementOffset(I)) -
          FieldSize;
      // Insert padding - this may include padding to increase the size of the
      // current field up to the ABI size (if the struct is not packed) as well
      // as padding to ensure that the next field starts at the right offset.
      pad(PadSize);
    }
  } else if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    for (unsigned I = 0; I < CDS->getNumElements(); I++)
      process(CDS->getElementAsConstant(I));
    pad(C, CDS->getNumElements());
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Look through bitcasts, which might not be able to be MCExpr'ized (e.g.
    // of vectors).
    if (CE->getOpcode() == Instruction::BitCast)
      return process(CE->getOperand(0));
    if (Size > 8) {
      // If the constant expression's size is greater than 64-bits, then we
      // have to emit the value in chunks. Try to constant fold the value and
      // emit it that way.
      Constant *New = ConstantFoldConstant(CE, DL);
      if (New != CE)
        return process(New);
    }
    lowerConstant(C);
  } else if (isa<GlobalVariable>(C) || isa<Function>(C)) {
    assert(Size == 8 && "global symbol with non 64-bit size?");
    lowerConstant(C);
  } else {
    llvm_unreachable("unhandled constant!");
  }
}

static bool isIgnoredIntrinsicGlobal(const GlobalVariable &GV) {
  if (GV.getName() == "llvm.used")
    return true;

  // Ignore debug and non-emitted data.  This handles llvm.compiler.used.
  if (GV.getSection() == "llvm.metadata")
    return true;

  // Skip globals only used as annotation strings by llvm.ptr.annotation.
  // These are metadata for the annotation intrinsic, not real data.
  if (GV.hasPrivateLinkage() && GV.isConstant() &&
      all_of(GV.users(), [](const User *U) {
        if (auto *CE = dyn_cast<ConstantExpr>(U))
          return all_of(CE->users(), [](const User *UU) {
            auto *CI = dyn_cast<CallInst>(UU);
            return CI && CI->getCalledFunction() &&
                   CI->getCalledFunction()->getIntrinsicID() ==
                       Intrinsic::ptr_annotation;
          });
        auto *CI = dyn_cast<CallInst>(U);
        return CI && CI->getCalledFunction() &&
               CI->getCalledFunction()->getIntrinsicID() ==
                   Intrinsic::ptr_annotation;
      }))
    return true;

  if (!GV.hasAppendingLinkage())
    return false;

  if (GV.getName() == "llvm.global_ctors")
    report_fatal_error(
        "llvm.global_ctors is not supported by the PISA backend");

  if (GV.getName() == "llvm.global_dtors")
    report_fatal_error(
        "llvm.global_ctors is not supported by the PISA backend");

  report_fatal_error("unknown special variable with appending linkage");
}

void PISAAsmPrinter::emitGlobalsAndFuncDecls(Module &M) {
  PISATargetStreamer &TS = getTargetStreamer();

  // emit header info
  // - we always emit in latest PISA syntax
  auto GetHdrTarget = [&]() -> SmallString<16> {
    return ST ? ST->getPISATargetName() : "";
  };
  PISA::HeaderDcl HD = {PISA::LatestPISAVersion, GetHdrTarget()};
  TS.emitHeader(HD);
  OutStreamer->addBlankLine();

  // Emit Module level function decl
  for (auto &F : M) {
    if (!F.isDeclaration() || F.isIntrinsic()) // avoid llvm builtins
      continue;

    PISA::FunctionDeclaration Dcl;

    collectFunctionDeclaration(Dcl, F);
    TS.emitFunctionDeclaration(Dcl);
    OutStreamer->addBlankLine();
  }

  // Translate global variables
  for (auto &GV : M.globals()) {
    if (isIgnoredIntrinsicGlobal(GV))
      continue;

    PISA::GlobalVariableDcl PGV;
    collectGlobalVariable(PGV, GV);
    TS.emitGlobalVariable(PGV);
  }
}

bool PISAAsmPrinter::doInitialization(Module &M) {
  GlobalsEmitted = false;
  return AsmPrinter::doInitialization(M);
}

bool PISAAsmPrinter::doFinalization(Module &M) {
  // If we did not emit any functions, then the global declarations have not
  // yet been emitted.
  if (!GlobalsEmitted) {
    emitGlobalsAndFuncDecls(M);
    GlobalsEmitted = true;
  }
  return AsmPrinter::doFinalization(M);
}

bool PISAAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<PISASubtarget>();
  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();

  if (!GlobalsEmitted) {
    emitGlobalsAndFuncDecls(*MF.getFunction().getParent());
    GlobalsEmitted = true;
  }

  PISA::RegManager Mgr{MF};
  RegMgr = &Mgr;

  return AsmPrinter::runOnMachineFunction(MF);
}

void PISAAsmPrinter::emitFunctionHeader() {
  const Function &F = MF->getFunction();

  auto *Section = getObjFileLowering().SectionForGlobal(&F, TM);
  MF->setSection(Section);
}

void PISAAsmPrinter::updateFuncParamIdxs(PISA::DataTypes &DTs) {
  // Update DataTypes records of RegStart for body register
  // declarations (vs the already-processed func param dcls)
  DTs.finalizeFuncParams();

  llvm::DenseMap<std::tuple</*NumElts=*/unsigned, /*BitWidth=*/unsigned,
                            /*Type=*/unsigned>,
                 /*Index=*/unsigned>
      ParamIdxs;

  auto &MRI = MF->getRegInfo();
  for (auto &[CurReg, Info] : RegMgr->mapping()) {
    // We are only trying to update indices for function parameters
    if (!(Info.Flags & PISA::RegManager::NoEmissionDef))
      continue;

    auto *RC = MRI.getRegClass(CurReg);
    unsigned BitWidth = TRI->getBitSizeFromRegClass(RC);
    unsigned NumElts = TRI->getNumEltsFromRegClass(RC);
    auto [It, Inserted] =
        ParamIdxs.try_emplace(std::make_tuple(NumElts, BitWidth, Info.Type), 0);
    RegMgr->setRegIdx(CurReg, It->second++);

    // Sanity check that all function parameter indexes are < the total
    // number of function parameters of that type (recorded in DTs)
    [[maybe_unused]] bool ValidIdx =
        Info.Idx < DTs.getInfo(NumElts, BitWidth, Info.Type).RegCounter;
    assert(ValidIdx && "function parameter index out of range!");
  }
}

void PISAAsmPrinter::collectRegDcls(PISA::RegDcls &Dcls) {
  auto &MRI = MF->getRegInfo();
  for (auto &[CurReg, Info] : RegMgr->mapping()) {
    if (Info.Flags & PISA::RegManager::NoEmissionDef)
      continue;
    auto *RC = MRI.getRegClass(CurReg);
    unsigned BitWidth = TRI->getBitSizeFromRegClass(RC);
    unsigned NumElts = TRI->getNumEltsFromRegClass(RC);
    TypeInfo &TI = DTs->emplaceInfo(NumElts, BitWidth, Info.Type);
    auto Bank = RegMgr->getRegBank(NumElts, BitWidth);
    const char *Prefix =
        RegMgr->getPrefixFromBank(static_cast<PISA::RegManager::RegBank>(Bank));
    Dcls.Regs[std::make_tuple(NumElts, BitWidth, Info.Type)].push_back(
        std::make_pair(Prefix, TI.RegCounter));
    RegMgr->setRegIdx(CurReg, TI.RegCounter);
    TI.RegCounter++;
  }
}

void PISAAsmPrinter::collectLocalVariableDcls(PISA::LocalVariableDcls &Dcls) {
  auto &MFI = MF->getFrameInfo();
  for (int Idx = MFI.getObjectIndexBegin(), EndIdx = MFI.getObjectIndexEnd();
       Idx != EndIdx; ++Idx) {
    if (MFI.isDeadObjectIndex(Idx))
      continue;
    // translation of 'alloca' creates a local 0-sized object within current
    // frame. Since PISA does not create real frame, omit such objects.
    if (!MFI.getObjectSize(Idx))
      continue;
    PISA::VariableDcl VarDecl;
    VarDecl.Linkage = PISA::LinkageTy::DEFAULT;
    switch (MFI.getStackID(Idx)) {
    case TargetStackID::Default:
      VarDecl.SS = PISA::StorageSpace::PRIVATE;
      break;
    case TargetStackID::PISAShared:
      VarDecl.SS = PISA::StorageSpace::SHARED;
      break;
    default:
      llvm_unreachable("unknown stack ID!");
    }

    VarDecl.Size = MFI.getObjectSize(Idx);
    VarDecl.Alignment = MFI.getObjectAlign(Idx);
    VarDecl.StackIndex = Idx;

    Dcls.Vars.push_back(std::move(VarDecl));
  }
}

PISA::LinkageTy PISAAsmPrinter::collectLinkage(const GlobalValue &V) {
  if (V.hasLocalLinkage())
    return PISA::LinkageTy::DEFAULT;

  // global variable linkage
  if (auto *GVar = dyn_cast<GlobalVariable>(&V)) {
    // External GV with no initializer must be .import. In llvm, global
    // variable definitions must be initialized. Though PISA allows
    // a GV definition with no initializer, we can safely determine the
    // linkage by having initializer or not here
    return GVar->hasInitializer() ? PISA::LinkageTy::EXPORT
                                  : PISA::LinkageTy::IMPORT;
  }

  // function variable linkage
  return V.isDeclaration() ? PISA::LinkageTy::IMPORT : PISA::LinkageTy::EXPORT;
}

static void collectIntelHostAccessMetadata(PISA::GlobalVariableDcl &PGV,
                                           const GlobalVariable &GV) {
  // !intel_host_access !{i32 <HostAccessQualifier>, !"<Name>"}
  // -> .host_access("Name")
  MDNode *MD = GV.getMetadata("intel_host_access");
  if (!MD || MD->getNumOperands() < 2)
    return;

  auto *NameMD = dyn_cast<MDString>(MD->getOperand(1));
  if (!NameMD)
    return;

  PGV.Dcl.HostAccessName = NameMD->getString().str();
}

void PISAAsmPrinter::collectGlobalVariable(PISA::GlobalVariableDcl &PGV,
                                           const GlobalVariable &GV) {
  auto &DL = GV.getParent()->getDataLayout();
  PGV.Dcl.Linkage = collectLinkage(GV);
  PGV.Dcl.SS =
      PISA::mapAddrSpaceToStorageSpace(GV.getType()->getAddressSpace());
  PGV.Dcl.Alignment = DL.getPreferredAlign(&GV);
  PGV.Dcl.Name = getSymbol(&GV)->getName();
  PGV.Dcl.Size = DL.getTypeAllocSize(GV.getValueType());
  PGV.Dcl.Section = GV.getSection();

  collectIntelHostAccessMetadata(PGV, GV);

  if (GV.hasInitializer() && !isa<UndefValue>(GV.getInitializer()))
    FlattenGlobal FG{GV.getInitializer(), PGV.Init, DL, *this};
}

void PISAAsmPrinter::collectFunctionDeclaration(PISA::FunctionDeclaration &Dcl,
                                                const Function &F) {
  assert(F.isDeclaration());
  assert(F.getCallingConv() != CallingConv::PISA_KERNEL);

  collectFunctionDirectiveAndName(Dcl.DN, F);
  for (auto &P : F.args()) {
    PISA::FunctionDeclParam Param;
    if (P.getType()->getScalarSizeInBits() == 1) {
      Param.Ty = LLT::integer(8);
    } else {
      Param.Ty = getLLTForType(*P.getType(), F.getParent()->getDataLayout());
    }
    Dcl.FunctionParams.push_back(Param);
  }
}

void PISAAsmPrinter::collectKernelParameters(PISA::FunctionSignature &Sig) {
  const PISAMachineFunctionInfo *MFInfo =
      MF->getInfo<PISAMachineFunctionInfo>();

  // print params
  auto &DL = MF->getFunction().getParent()->getDataLayout();
  for (unsigned Index = 0; Index < MF->getFunction().arg_size(); ++Index) {
    auto [Size, IsByRef] = MFInfo->getArgInfo(Index);
    PISA::KernelParameter Param;
    Param.Size = Size;
    auto *ArgTy = MF->getFunction().getArg(Index)->getType()->getScalarType();
    auto Align = alignTo(PowerOf2Ceil(DL.getABITypeAlign(ArgTy).value()), 4);
    if (Align != 8) {
      // Kernel parameters are aligned to 8 bytes by default.
      Param.Align = Align;
    }
    if (ArgTy->isPointerTy() && !IsByRef) {
      auto AS = ArgTy->getPointerAddressSpace();
      if ((AS == (unsigned)PISAAS::AddressSpace::CONSTANT) ||
          (AS == (unsigned)PISAAS::AddressSpace::GLOBAL) ||
          (AS == (unsigned)PISAAS::AddressSpace::SHARED))
        Param.AS = ArgTy->getPointerAddressSpace();
      if (auto PtrAlign = MF->getFunction().getParamAlign(Index))
        Param.PtrAlign = PtrAlign->value();
    }

    // Read OpenCL kernel arg metadata (emitted by the OpenCL frontend with
    // -cl-kernel-arg-info).
    const Function &F = MF->getFunction();
    if (MDNode *MD = F.getMetadata("kernel_arg_name"))
      if (Index < MD->getNumOperands())
        if (auto *S = dyn_cast<MDString>(MD->getOperand(Index)))
          if (!S->getString().empty())
            Param.ArgName = S->getString().str();

    Sig.KernelParams.push_back(std::move(Param));
  }
}

void PISAAsmPrinter::collectFunctionParameters(PISA::FunctionSignature &Sig) {
  llvm::SmallVector<const MachineInstr *, 8> FuncParamInsts;
  for (auto &MBB : *MF) {
    // FunctionParam must be contiguous and in the same BB
    // Find the iterator of the first FunctionParam inst and iterate from it
    // to collect all FunctionParam insts
    auto MIIt = find_if(
        MBB, [&](MachineInstr &MI) { return TII->isFunctionParamInstr(MI); });

    for (; MIIt != MBB.end(); ++MIIt) {
      if (!TII->isFunctionParamInstr(*MIIt))
        break;
      FuncParamInsts.push_back(&*MIIt);
    }
  }

  // Sort FunctionParam by param index
  llvm::sort(FuncParamInsts, [](const MachineInstr *L, const MachineInstr *R) {
    return L->getOperand(1).getImm() < R->getOperand(1).getImm();
  });

  // Collect params
  for (auto *MI : FuncParamInsts) {
    const MachineOperand &MO = MI->getOperand(0);
    assert(MO.getSubReg() == 0 && "no swizzle allowed on args!");
    const auto *RC = MF->getRegInfo().getRegClass(MO.getReg());
    PISA::FunctionParameter Param;
    unsigned NumElts = TRI->getNumEltsFromRegClass(RC);
    unsigned EltSize = TRI->getBitSizeFromRegClass(RC);
    TypeInfo &TI = DTs->emplaceInfo(NumElts, EltSize, PISA::RegEncoder::REG);
    Param.Ty = TI.Ty;
    Param.Prefix = TI.Prefix;
    Param.Idx = TI.RegCounter++;
    Sig.FunctionParams.push_back(std::move(Param));
  }
}

static std::string getNameFromType(Type *Ty, bool IsSigned) {
  std::string Name = "unknown";
  switch (Ty->getTypeID()) {
  default:
    llvm_unreachable("unsupported type");
    break;
  case Type::IntegerTyID: {
    switch (Ty->getIntegerBitWidth()) {
    default:
      llvm_unreachable("unsupported integer type");
      break;
    case 8:
      Name = IsSigned ? "char" : "uchar";
      break;
    case 16:
      Name = IsSigned ? "short" : "ushort";
      break;
    case 32:
      Name = IsSigned ? "int" : "uint";
      break;
    case 64:
      Name = IsSigned ? "long" : "ulong";
      break;
    }
  } break;
  case Type::HalfTyID:
    Name = "half";
    break;
  case Type::FloatTyID:
    Name = "float";
    break;
  case Type::DoubleTyID:
    Name = "double";
    break;
  case Type::FixedVectorTyID: {
    auto *VecTy = cast<FixedVectorType>(Ty);
    Name = getNameFromType(VecTy->getElementType(), IsSigned) +
           std::to_string(VecTy->getNumElements());
  } break;
  }
  return Name;
}

void PISAAsmPrinter::collectFunctionDirectiveAndName(
    PISA::FunctionDirectiveAndName &DN, const Function &F) {

  DN.CC = F.getCallingConv();
  if (DN.CC != CallingConv::PISA_KERNEL)
    DN.Linkage = collectLinkage(F);
  DN.Name = getSymbol(&F)->getName();
  if (DN.CC != CallingConv::PISA_KERNEL) {
    Type *RetType = F.getReturnType();
    if (!RetType->isVoidTy()) {
      if (RetType->getScalarSizeInBits() == 1) {
        DN.RetLLT = LLT::integer(8);
      } else {
        DN.RetLLT =
            llvm::getLLTForType(*RetType, F.getParent()->getDataLayout());
      }
    }
  }

  std::vector<std::pair<StringRef, PISA::KernelAttributeType>>
      AvailableKernelMetadataNodeTypes = {
          {"reqd_work_group_size",
           PISA::KernelAttributeType::REQD_WORK_GROUP_SIZE},
          {"vec_type_hint", PISA::KernelAttributeType::VEC_TYPE_HINT}};

  for (auto [MetadataName, EnumVal] : AvailableKernelMetadataNodeTypes) {
    MDNode *Node = dyn_cast_or_null<MDNode>(F.getMetadata(MetadataName));
    if (!Node)
      continue;
    auto &KernelAttr = DN.KernelAttrs.emplace_back();
    KernelAttr.KernelAttrType = EnumVal;
    switch (EnumVal) {
    case llvm::PISA::KernelAttributeType::REQD_WORK_GROUP_SIZE: {
      KernelAttr.KernelAttrValues.emplace<std::vector<uint32_t>>();
      std::transform(Node->op_begin(), Node->op_end(),
                     std::back_inserter(std::get<std::vector<uint32_t>>(
                         KernelAttr.KernelAttrValues)),
                     [](const MDOperand &Operand) -> uint32_t {
                       const ValueAsMetadata *OperandAsVal =
                           cast<ValueAsMetadata>(Operand);
                       ConstantInt *OperandVal =
                           cast<ConstantInt>(OperandAsVal->getValue());
                       return static_cast<uint32_t>(OperandVal->getZExtValue());
                     });
    } break;
    case llvm::PISA::KernelAttributeType::VEC_TYPE_HINT: {
      Metadata *Op0 = Node->getOperand(0);
      Metadata *Op1 = Node->getOperand(1);
      ConstantInt *CI =
          cast<ConstantInt>(cast<ValueAsMetadata>(Op1)->getValue());
      auto TypeName =
          getNameFromType(cast<ValueAsMetadata>(Op0)->getType(), CI->isOne());
      KernelAttr.KernelAttrValues.emplace<std::string>(TypeName);
    } break;
    }
  }
}

void PISAAsmPrinter::collectFunctionSignature(PISA::FunctionSignature &Sig) {
  Function &F = MF->getFunction();
  collectFunctionDirectiveAndName(Sig.DN, F);
  if (Sig.DN.CC == CallingConv::PISA_KERNEL)
    collectKernelParameters(Sig);
  else
    collectFunctionParameters(Sig);
}

void PISAAsmPrinter::emitFunctionBodyStart() {
  PISATargetStreamer &TS = getTargetStreamer();
  PISA::FunctionSignature Sig;
  PISA::DataTypes FuncDTs;
  DTs = &FuncDTs;
  // Collect function signature.
  collectFunctionSignature(Sig);
  TS.emitFunctionSignature(Sig);
  updateFuncParamIdxs(*DTs);
  // Emit the function body start.
  TS.emitFuncBodyStart();
  // Collect local registers.
  PISA::RegDcls Regs;
  collectRegDcls(Regs);
  TS.emitRegDcls(Regs, *DTs);
  // Collect local variables.
  PISA::LocalVariableDcls Vars;
  collectLocalVariableDcls(Vars);
  TS.emitLocalVariableDcls(Vars);
}

void PISAAsmPrinter::emitFunctionBodyEnd() {
  PISATargetStreamer &TS = getTargetStreamer();
  // Emit the function body end.
  TS.emitFuncBodyEnd();
}

void PISAAsmPrinter::emitEndOfAsmFile(llvm::Module &) {}

void PISAAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                  raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    auto Reg = MO.getReg();
    if (Reg.isPhysical())
      O << PISAInstPrinter::getRegisterName(Reg);
    else {
      O << getVirtualRegisterName(Reg);
      O << TRI->getSwizzleName(MO.getSubReg());
    }
  } break;

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

bool PISAAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                     const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Invalid instruction - PISA does not have special
                 // modifiers

  printOperand(MI, OpNo, O);
  return false;
}

std::string PISAAsmPrinter::getVirtualRegisterName(Register R) const {
  auto &MRI = MF->getRegInfo();
  const auto *RC = MRI.getRegClass(R);

  std::string Name;
  raw_string_ostream O(Name);

  unsigned NumElts = TRI->getNumEltsFromRegClass(RC);
  unsigned EltSize = TRI->getBitSizeFromRegClass(RC);
  RegEncoder::RegBank Bank = RegMgr->getRegBank(NumElts, EltSize);
  O << RegMgr->getPrefixFromBank(Bank) << RegMgr->getRegIdx(R);
  return Name;
}

void PISAAsmPrinter::outputInstruction(const MachineInstr *MI) {
  PISAMCInstLower MCInstLowering{OutContext, *TRI, *RegMgr, *this};
  PISAMCInst Inst;
  MCInstLowering.lower(MI, Inst);
  if (MI->getOpcode() == PISA::DBG_VALUE)
    return;
  OutStreamer->emitInstruction(Inst, *OutContext.getSubtargetInfo());
}

void PISAAsmPrinter::emitInstruction(const MachineInstr *MI) {
  PISA_MC::verifyInstructionPredicates(MI->getOpcode(),
                                       getSubtargetInfo().getFeatureBits());

  if (!TII->isNoEmissionInstr(*MI))
    outputInstruction(MI);
}

// Force static initialization.
// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializePISAAsmPrinter() {
  RegisterAsmPrinter<PISAAsmPrinter> Y(getThePISATarget());
}
