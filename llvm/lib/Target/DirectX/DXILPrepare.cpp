//===- DXILPrepare.cpp - Prepare LLVM Module for DXIL encoding ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains pases and utilities to convert a modern LLVM
/// module into a module compatible with the LLVM 3.7-based DirectX Intermediate
/// Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "DirectXIRPasses/PointerTypeAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/VersionTuple.h"

#define DEBUG_TYPE "dxil-prepare"

using namespace llvm;
using namespace llvm::dxil;

namespace {

constexpr bool isValidForDXIL(Attribute::AttrKind Attr) {
  return is_contained({Attribute::Alignment,
                       Attribute::AlwaysInline,
                       Attribute::Builtin,
                       Attribute::ByVal,
                       Attribute::InAlloca,
                       Attribute::Cold,
                       Attribute::Convergent,
                       Attribute::InlineHint,
                       Attribute::InReg,
                       Attribute::JumpTable,
                       Attribute::MinSize,
                       Attribute::Naked,
                       Attribute::Nest,
                       Attribute::NoAlias,
                       Attribute::NoBuiltin,
                       Attribute::NoDuplicate,
                       Attribute::NoImplicitFloat,
                       Attribute::NoInline,
                       Attribute::NonLazyBind,
                       Attribute::NonNull,
                       Attribute::Dereferenceable,
                       Attribute::DereferenceableOrNull,
                       Attribute::Memory,
                       Attribute::NoRedZone,
                       Attribute::NoReturn,
                       Attribute::NoUnwind,
                       Attribute::OptimizeForSize,
                       Attribute::OptimizeNone,
                       Attribute::ReadNone,
                       Attribute::ReadOnly,
                       Attribute::Returned,
                       Attribute::ReturnsTwice,
                       Attribute::SExt,
                       Attribute::StackAlignment,
                       Attribute::StackProtect,
                       Attribute::StackProtectReq,
                       Attribute::StackProtectStrong,
                       Attribute::SafeStack,
                       Attribute::StructRet,
                       Attribute::SanitizeAddress,
                       Attribute::SanitizeThread,
                       Attribute::SanitizeMemory,
                       Attribute::UWTable,
                       Attribute::ZExt},
                      Attr);
}

static void collectDeadStringAttrs(AttributeMask &DeadAttrs, AttributeSet &&AS,
                                   const StringSet<> &LiveKeys,
                                   bool AllowExperimental) {
  for (auto &Attr : AS) {
    if (!Attr.isStringAttribute())
      continue;
    StringRef Key = Attr.getKindAsString();
    if (LiveKeys.contains(Key))
      continue;
    if (AllowExperimental && Key.starts_with("exp-"))
      continue;
    DeadAttrs.addAttribute(Key);
  }
}

static void removeStringFunctionAttributes(Function &F,
                                           bool AllowExperimental) {
  AttributeList Attrs = F.getAttributes();
  const StringSet<> LiveKeys = {"waveops-include-helper-lanes",
                                "fp32-denorm-mode"};
  // Collect DeadKeys in FnAttrs.
  AttributeMask DeadAttrs;
  collectDeadStringAttrs(DeadAttrs, Attrs.getFnAttrs(), LiveKeys,
                         AllowExperimental);
  collectDeadStringAttrs(DeadAttrs, Attrs.getRetAttrs(), LiveKeys,
                         AllowExperimental);

  F.removeFnAttrs(DeadAttrs);
  F.removeRetAttrs(DeadAttrs);
}

static void cleanModuleFlags(Module &M) {
  NamedMDNode *MDFlags = M.getModuleFlagsMetadata();
  if (!MDFlags)
    return;

  SmallVector<llvm::Module::ModuleFlagEntry> FlagEntries;
  M.getModuleFlagsMetadata(FlagEntries);
  bool Updated = false;
  for (auto &Flag : FlagEntries) {
    // llvm 3.7 only supports behavior up to AppendUnique.
    if (Flag.Behavior <= Module::ModFlagBehavior::AppendUnique)
      continue;
    Flag.Behavior = Module::ModFlagBehavior::Warning;
    Updated = true;
  }

  if (!Updated)
    return;

  MDFlags->eraseFromParent();

  for (auto &Flag : FlagEntries)
    M.addModuleFlag(Flag.Behavior, Flag.Key->getString(), Flag.Val);
}

class DXILPrepareModule : public ModulePass {

  static Value *maybeGenerateBitcast(IRBuilder<> &Builder,
                                     PointerTypeMap &PointerTypes,
                                     Instruction &Inst, Value *Operand,
                                     Type *Ty) {
    // Omit bitcasts if the incoming value matches the instruction type.
    auto It = PointerTypes.find(Operand);
    if (It != PointerTypes.end()) {
      auto *OpTy = cast<TypedPointerType>(It->second)->getElementType();
      if (OpTy == Ty)
        return nullptr;
    }

    Type *ValTy = Operand->getType();
    // Also omit the bitcast for matching global array types
    if (auto *GlobalVar = dyn_cast<GlobalVariable>(Operand))
      ValTy = GlobalVar->getValueType();

    if (auto *AI = dyn_cast<AllocaInst>(Operand))
      ValTy = AI->getAllocatedType();

    if (auto *ArrTy = dyn_cast<ArrayType>(ValTy)) {
      Type *ElTy = ArrTy->getElementType();
      if (ElTy == Ty)
        return nullptr;
    }

    // finally, drill down GEP instructions until we get the array
    // that is being accessed, and compare element types
    if (ConstantExpr *GEPInstr = dyn_cast<ConstantExpr>(Operand)) {
      while (GEPInstr->getOpcode() == Instruction::GetElementPtr) {
        Value *OpArg = GEPInstr->getOperand(0);
        if (ConstantExpr *NewGEPInstr = dyn_cast<ConstantExpr>(OpArg)) {
          GEPInstr = NewGEPInstr;
          continue;
        }

        if (auto *GlobalVar = dyn_cast<GlobalVariable>(OpArg))
          ValTy = GlobalVar->getValueType();
        if (auto *AI = dyn_cast<AllocaInst>(Operand))
          ValTy = AI->getAllocatedType();
        if (auto *ArrTy = dyn_cast<ArrayType>(ValTy)) {
          Type *ElTy = ArrTy->getElementType();
          if (ElTy == Ty)
            return nullptr;
        }
        break;
      }
    }

    // Insert bitcasts where we are removing the instruction.
    Builder.SetInsertPoint(&Inst);
    // This code only gets hit in opaque-pointer mode, so the type of the
    // pointer doesn't matter.
    PointerType *PtrTy = cast<PointerType>(Operand->getType());
    return Builder.Insert(
        CastInst::Create(Instruction::BitCast, Operand,
                         Builder.getPtrTy(PtrTy->getAddressSpace())));
  }

  static std::array<unsigned, 6> getCompatibleInstructionMDs(llvm::Module &M) {
    return {M.getMDKindID("dx.nonuniform"),
            M.getMDKindID("dx.controlflow.hints"),
            M.getMDKindID("dx.precise"),
            llvm::LLVMContext::MD_range,
            llvm::LLVMContext::MD_alias_scope,
            llvm::LLVMContext::MD_noalias};
  }

public:
  bool runOnModule(Module &M) override {
    PointerTypeMap PointerTypes = PointerTypeAnalysis::run(M);
    AttributeMask AttrMask;
    for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
         I = Attribute::AttrKind(I + 1)) {
      if (!isValidForDXIL(I))
        AttrMask.addAttribute(I);
    }

    const dxil::ModuleMetadataInfo MetadataInfo =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();
    VersionTuple ValVer = MetadataInfo.ValidatorVersion;
    bool SkipValidation = ValVer.getMajor() == 0 && ValVer.getMinor() == 0;

    // construct allowlist of valid metadata node kinds
    std::array<unsigned, 6> DXILCompatibleMDs = getCompatibleInstructionMDs(M);

    for (auto &F : M.functions()) {
      F.removeFnAttrs(AttrMask);
      F.removeRetAttrs(AttrMask);
      // Only remove string attributes if we are not skipping validation.
      // This will reserve the experimental attributes when validation version
      // is 0.0 for experiment mode.
      removeStringFunctionAttributes(F, SkipValidation);
      for (size_t Idx = 0, End = F.arg_size(); Idx < End; ++Idx)
        F.removeParamAttrs(Idx, AttrMask);

      for (auto &BB : F) {
        IRBuilder<> Builder(&BB);
        for (auto &I : make_early_inc_range(BB)) {

          I.dropUnknownNonDebugMetadata(DXILCompatibleMDs);

          // Emtting NoOp bitcast instructions allows the ValueEnumerator to be
          // unmodified as it reserves instruction IDs during contruction.
          if (auto LI = dyn_cast<LoadInst>(&I)) {
            if (Value *NoOpBitcast = maybeGenerateBitcast(
                    Builder, PointerTypes, I, LI->getPointerOperand(),
                    LI->getType())) {
              LI->replaceAllUsesWith(
                  Builder.CreateLoad(LI->getType(), NoOpBitcast));
              LI->eraseFromParent();
            }
            continue;
          }
          if (auto SI = dyn_cast<StoreInst>(&I)) {
            if (Value *NoOpBitcast = maybeGenerateBitcast(
                    Builder, PointerTypes, I, SI->getPointerOperand(),
                    SI->getValueOperand()->getType())) {

              SI->replaceAllUsesWith(
                  Builder.CreateStore(SI->getValueOperand(), NoOpBitcast));
              SI->eraseFromParent();
            }
            continue;
          }
          if (auto GEP = dyn_cast<GetElementPtrInst>(&I)) {
            if (Value *NoOpBitcast = maybeGenerateBitcast(
                    Builder, PointerTypes, I, GEP->getPointerOperand(),
                    GEP->getSourceElementType()))
              GEP->setOperand(0, NoOpBitcast);
            continue;
          }
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            CB->removeFnAttrs(AttrMask);
            CB->removeRetAttrs(AttrMask);
            for (size_t Idx = 0, End = CB->arg_size(); Idx < End; ++Idx)
              CB->removeParamAttrs(Idx, AttrMask);
            continue;
          }
        }
      }
    }
    // Remove flags not for DXIL.
    cleanModuleFlags(M);

    // dx.rootsignatures will have been parsed from its metadata form as its
    // binary form as part of the RootSignatureAnalysisWrapper, so safely
    // remove it as it is not recognized in DXIL
    if (NamedMDNode *RootSignature = M.getNamedMetadata("dx.rootsignatures"))
      RootSignature->eraseFromParent();

    return true;
  }

  DXILPrepareModule() : ModulePass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addRequired<RootSignatureAnalysisWrapper>();
    AU.addPreserved<RootSignatureAnalysisWrapper>();
    AU.addPreserved<ShaderFlagsAnalysisWrapper>();
    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
  }
  static char ID; // Pass identification.
};
char DXILPrepareModule::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILPrepareModule, DEBUG_TYPE, "DXIL Prepare Module",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RootSignatureAnalysisWrapper)
INITIALIZE_PASS_END(DXILPrepareModule, DEBUG_TYPE, "DXIL Prepare Module", false,
                    false)

ModulePass *llvm::createDXILPrepareModulePass() {
  return new DXILPrepareModule();
}
