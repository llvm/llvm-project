//===- MetalAIRSystemValues.cpp - AIR system-value lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalAIRSystemValues.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#include <cstdio>

using namespace llvm;

#define DEBUG_TYPE "metal-air-system-values"

// AIR system value intrinsic-name prefixes.
static constexpr StringLiteral kCallPid("air.threadgroup_position_in_grid");
static constexpr StringLiteral kCallTid("air.thread_position_in_grid");
static constexpr StringLiteral kCallTidTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral kCallSimdlane("air.thread_index_in_simdgroup");
static constexpr StringLiteral kCallNumProg("air.threadgroups_per_grid");

// AIR metadata-tag strings.
static constexpr StringLiteral
    kMDThreadPositionInGrid("air.thread_position_in_grid");
static constexpr StringLiteral
    kMDThreadgroupPositionInGrid("air.threadgroup_position_in_grid");
static constexpr StringLiteral
    kMDThreadPositionInTG("air.thread_position_in_threadgroup");
static constexpr StringLiteral
    kMDThreadIndexInSimdgroup("air.thread_index_in_simdgroup");
static constexpr StringLiteral
    kMDThreadgroupsPerGrid("air.threadgroups_per_grid");

static constexpr StringLiteral kMDBuffer("air.buffer");
static constexpr StringLiteral kMDLocationIndex("air.location_index");
static constexpr StringLiteral kMDRead("air.read");
static constexpr StringLiteral kMDReadWrite("air.read_write");
static constexpr StringLiteral kMDAddressSpace("air.address_space");
static constexpr StringLiteral kMDArgTypeSize("air.arg_type_size");
static constexpr StringLiteral kMDArgTypeAlignSize("air.arg_type_align_size");
static constexpr StringLiteral kMDArgTypeName("air.arg_type_name");
static constexpr StringLiteral kMDArgName("air.arg_name");

static constexpr StringLiteral kNMDKernel("air.kernel");
static constexpr StringLiteral kNMDVersion("air.version");
static constexpr StringLiteral kNMDLanguageVersion("air.language_version");

namespace {
struct SysValParam {
  StringRef IntrinsicName;
  StringRef ParamName;
  bool IsVector; // true = <3 x i32>, false = i32
  const char *Dims[3];
};
} // namespace

static const SysValParam kSysVals[] = {
    {kCallPid, "pid", true, {"pid_x", "pid_y", "pid_z"}},
    {kCallTid, "tid", true, {"tid_x", "tid_y", "tid_z"}},
    {kCallTidTG, "tidtg", true, {"tidtg_x", "tidtg_y", "tidtg_z"}},
    {kCallSimdlane, "simdlane", false, {"simdlane", "", ""}},
    {kCallNumProg, "numprog", true, {"numprog_x", "numprog_y", "numprog_z"}},
};

static Function *findDeclByPrefix(Module &M, StringRef Prefix) {
  for (Function &F : M)
    if (F.isDeclaration() && F.getName().starts_with(Prefix))
      return &F;
  return nullptr;
}

static bool airSystemValues(Module &M) {
  auto &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *Vec3 = FixedVectorType::get(I32, 3);

  bool Changed = false;

  SmallVector<Function *, 4> Funcs;
  for (Function &F : M)
    if (!F.isDeclaration())
      Funcs.push_back(&F);

  for (Function *FPtr : Funcs) {
    Function &F = *FPtr;

    struct UsedSysVal {
      const SysValParam *Info;
      SmallVector<CallInst *, 4> Calls;
    };
    SmallVector<UsedSysVal, 5> UsedSysVals;

    for (const SysValParam &SV : kSysVals) {
      Function *Decl = findDeclByPrefix(M, SV.IntrinsicName);
      if (!Decl)
        continue;
      SmallVector<CallInst *, 4> Calls;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (CI->getCalledFunction() == Decl)
              Calls.push_back(CI);
      if (!Calls.empty())
        UsedSysVals.push_back({&SV, std::move(Calls)});
    }

    if (UsedSysVals.empty())
      continue;

    // Build a new function type with extra params at the end.
    auto *OldFTy = F.getFunctionType();
    SmallVector<Type *, 8> NewParamTypes(OldFTy->params());
    for (const UsedSysVal &USV : UsedSysVals)
      NewParamTypes.push_back(USV.Info->IsVector ? Vec3 : I32);

    auto *NewFTy = FunctionType::get(OldFTy->getReturnType(), NewParamTypes,
                                     OldFTy->isVarArg());

    auto *NewF =
        Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", &M);
    NewF->copyAttributesFrom(&F);
    NewF->splice(NewF->begin(), &F);

    auto NewArgIt = NewF->arg_begin();
    for (Argument &OldArg : F.args()) {
      NewArgIt->setName(OldArg.getName());
      OldArg.replaceAllUsesWith(&*NewArgIt);
      ++NewArgIt;
    }

    std::string FName = F.getName().str();
    F.eraseFromParent();
    NewF->setName(FName);

    BasicBlock &EntryBB = NewF->getEntryBlock();

    for (UsedSysVal &USV : UsedSysVals) {
      Argument *SysArg = &*NewArgIt++;
      SysArg->setName(USV.Info->ParamName);

      if (USV.Info->IsVector) {
        Value *Components[3];
        for (int K = 0; K < 3; ++K) {
          IRBuilder<> B(&*EntryBB.begin());
          Components[K] = B.CreateExtractElement(
              SysArg, ConstantInt::get(I32, K), USV.Info->Dims[K]);
        }
        for (CallInst *CI : USV.Calls) {
          SmallVector<ExtractValueInst *, 4> Extracts;
          for (User *U : CI->users())
            if (auto *EV = dyn_cast<ExtractValueInst>(U))
              Extracts.push_back(EV);
          for (ExtractValueInst *EV : Extracts) {
            unsigned Idx = EV->getIndices()[0];
            EV->replaceAllUsesWith(Components[std::min(Idx, 2u)]);
            EV->eraseFromParent();
          }
          if (!CI->use_empty())
            CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
          CI->eraseFromParent();
        }
      } else {
        for (CallInst *CI : USV.Calls) {
          CI->replaceAllUsesWith(SysArg);
          CI->eraseFromParent();
        }
      }
    }

    Changed = true;
  }

  // Remove unused intrinsic declarations.
  for (const SysValParam &SV : kSysVals) {
    if (auto *F = findDeclByPrefix(M, SV.IntrinsicName))
      if (F->use_empty())
        F->eraseFromParent();
  }

  // Emit !air.kernel metadata (always; skip if pre-baked).
  if (!M.getNamedMetadata(kNMDKernel)) {
    auto *KernelMD = M.getOrInsertNamedMetadata(kNMDKernel);

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      SmallVector<Metadata *, 16> ParamNodes;
      unsigned ArgIdx = 0;
      auto *FTy = F.getFunctionType();

      // Buffer params (device AS=1, constant AS=2).
      for (unsigned I = 0; I < FTy->getNumParams(); ++I) {
        Type *ParamTy = FTy->getParamType(I);
        if (!ParamTy->isPointerTy())
          continue;
        unsigned AS = cast<PointerType>(ParamTy)->getAddressSpace();
        if (AS != 1 && AS != 2)
          continue;

        StringRef ArgName = F.getArg(I)->getName();
        char NameBuf[16];
        if (ArgName.empty()) {
          std::snprintf(NameBuf, sizeof(NameBuf), "%u", I);
          ArgName = NameBuf;
        }
        StringRef AccessMode = (AS == 2) ? kMDRead : kMDReadWrite;
        ParamNodes.push_back(MDNode::get(
            Ctx,
            {ConstantAsMetadata::get(ConstantInt::get(I32, I)),
             MDString::get(Ctx, kMDBuffer),
             MDString::get(Ctx, kMDLocationIndex),
             ConstantAsMetadata::get(ConstantInt::get(I32, ArgIdx)),
             ConstantAsMetadata::get(ConstantInt::get(I32, 1)),
             MDString::get(Ctx, AccessMode),
             MDString::get(Ctx, kMDAddressSpace),
             ConstantAsMetadata::get(ConstantInt::get(I32, AS)),
             MDString::get(Ctx, kMDArgTypeSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
             MDString::get(Ctx, kMDArgTypeAlignSize),
             ConstantAsMetadata::get(ConstantInt::get(I32, 4)),
             MDString::get(Ctx, kMDArgTypeName), MDString::get(Ctx, "float"),
             MDString::get(Ctx, kMDArgName), MDString::get(Ctx, ArgName)}));
        ArgIdx++;
      }

      // System value params (non-pointer, prefix-matched).
      for (Argument &Arg : F.args()) {
        Type *ArgTy = Arg.getType();
        if (ArgTy->isPointerTy())
          continue;
        StringRef Name = Arg.getName();
        StringRef AirAttr;
        if (Name.starts_with("pid"))
          AirAttr = kMDThreadgroupPositionInGrid;
        else if (Name.starts_with("tidtg"))
          AirAttr = kMDThreadPositionInTG;
        else if (Name.starts_with("tid"))
          AirAttr = kMDThreadPositionInGrid;
        else if (Name.starts_with("simdlane"))
          AirAttr = kMDThreadIndexInSimdgroup;
        else if (Name.starts_with("numprog"))
          AirAttr = kMDThreadgroupsPerGrid;
        else
          continue;

        StringRef TypeName = (Name == "simdlane") ? "uint" : "uint3";
        ParamNodes.push_back(MDNode::get(
            Ctx,
            {ConstantAsMetadata::get(ConstantInt::get(I32, Arg.getArgNo())),
             MDString::get(Ctx, AirAttr), MDString::get(Ctx, kMDArgTypeName),
             MDString::get(Ctx, TypeName), MDString::get(Ctx, kMDArgName),
             MDString::get(Ctx, Name)}));
      }

      auto *EmptyNode = MDNode::get(Ctx, {});
      auto *ParamsNode = MDNode::get(Ctx, ParamNodes);
      KernelMD->addOperand(
          MDNode::get(Ctx, {ValueAsMetadata::get(&F), EmptyNode, ParamsNode}));
    }

    Changed = true;
  }

  // Version metadata.
  auto *VerMD = M.getOrInsertNamedMetadata(kNMDVersion);
  if (VerMD->getNumOperands() == 0) {
    VerMD->addOperand(
        MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, 2)),
                          ConstantAsMetadata::get(ConstantInt::get(I32, 8)),
                          ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    Changed = true;
  }

  // Language version metadata.
  if (!M.getNamedMetadata(kNMDLanguageVersion)) {
    auto *LangMD = M.getOrInsertNamedMetadata(kNMDLanguageVersion);
    LangMD->addOperand(
        MDNode::get(Ctx, {MDString::get(Ctx, "Metal"),
                          ConstantAsMetadata::get(ConstantInt::get(I32, 3)),
                          ConstantAsMetadata::get(ConstantInt::get(I32, 2)),
                          ConstantAsMetadata::get(ConstantInt::get(I32, 0))}));
    Changed = true;
  }

  // Module flags: air.max_* limits.
  if (!M.getModuleFlag("air.max_device_buffers")) {
    M.addModuleFlag(Module::Max, "air.max_device_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_constant_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_threadgroup_buffers",
                    ConstantInt::get(I32, 31));
    M.addModuleFlag(Module::Max, "air.max_textures",
                    ConstantInt::get(I32, 128));
    M.addModuleFlag(Module::Max, "air.max_read_write_textures",
                    ConstantInt::get(I32, 8));
    M.addModuleFlag(Module::Max, "air.max_samplers", ConstantInt::get(I32, 16));
  }

  return Changed;
}

PreservedAnalyses MetalAIRSystemValuesPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return airSystemValues(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool MetalAIRSystemValuesLegacy::runOnModule(Module &M) {
  return airSystemValues(M);
}

char MetalAIRSystemValuesLegacy::ID = 0;

INITIALIZE_PASS(MetalAIRSystemValuesLegacy, DEBUG_TYPE,
                "Metal AIR System Values", false, false)

ModulePass *llvm::createMetalAIRSystemValuesLegacyPass() {
  return new MetalAIRSystemValuesLegacy();
}
