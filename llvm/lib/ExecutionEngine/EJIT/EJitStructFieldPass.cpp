//===-- EJitStructFieldPass.cpp - JIT Constant Substitution ---------------===//

#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include <cstring>

using namespace llvm;
using namespace llvm::ejit;

namespace {

struct GVPeriodInfo {
  std::string periodName;
  bool isArray;
  size_t arraySize;
};

static void buildGVPeriodMap(
    Module &M,
    DenseMap<const GlobalVariable *, GVPeriodInfo> &gvMap) {
  for (GlobalVariable &GV : M.globals()) {
    MDNode *MD = GV.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;

    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 2)
        continue;

      auto *Tag = dyn_cast<MDString>(Sub->getOperand(0));
      if (!Tag)
        continue;

      if (Tag->getString() == TAG_EJIT_PERIOD_ARR) {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        size_t sz = 0;
        if (Sub->getNumOperands() >= 3)
          if (auto *CI = mdconst::dyn_extract<ConstantInt>(Sub->getOperand(2)))
            sz = CI->getZExtValue();
        if (PN)
          gvMap[&GV] = {PN->getString().str(), true, sz};
      } else if (Tag->getString() == TAG_EJIT_PERIOD) {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        std::string pn = PN ? PN->getString().str() : "";
        gvMap[&GV] = {pn, false, 0};
      }
    }
  }
}

static const GlobalVariable *findRootGV(const Value *V) {
  V = V->stripPointerCasts();
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    return GV;
  if (auto *GEP = dyn_cast<GEPOperator>(V))
    return findRootGV(GEP->getPointerOperand());
  return nullptr;
}

/// Accumulate byte offset for constant GEP indices using DataLayout.
static std::optional<uint64_t>
accumulateConstantOffset(const DataLayout &DL,
                         const GEPOperator *GEP) {
  APInt offset(DL.getPointerSizeInBits(0), 0);

  for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; ++I) {
    Value *Idx = *I;
    auto *CI = dyn_cast<ConstantInt>(Idx);
    if (!CI)
      return std::nullopt;

    if (I == GEP->idx_begin()) {
      offset += CI->getValue().sextOrTrunc(offset.getBitWidth()) *
                APInt(offset.getBitWidth(),
                      DL.getTypeAllocSize(GEP->getSourceElementType()));
    } else {
      SmallVector<Value *, 4> IdxList;
      for (auto J = GEP->idx_begin(); J != I + 1; ++J)
        IdxList.push_back(*J);
      int64_t typeOffset = DL.getIndexedOffsetInType(
          GEP->getSourceElementType(), IdxList);
      offset += APInt(offset.getBitWidth(), typeOffset);
    }
  }
  return offset.getZExtValue();
}

static Constant *createConstantFromMemory(const void *addr, Type *Ty,
                                          const DataLayout &DL) {
  LLVMContext &Ctx = Ty->getContext();
  unsigned byteSize = DL.getTypeStoreSize(Ty);

  if (Ty->isIntegerTy()) {
    APInt val(byteSize * 8, 0);
    if (byteSize <= 8) {
      uint64_t raw = 0;
      std::memcpy(&raw, addr, byteSize);
      val = APInt(byteSize * 8, raw);
    }
    return ConstantInt::get(Ty, val);
  }
  if (Ty->isFloatTy()) {
    float v;
    std::memcpy(&v, addr, sizeof(v));
    return ConstantFP::get(Ty, v);
  }
  if (Ty->isDoubleTy()) {
    double v;
    std::memcpy(&v, addr, sizeof(v));
    return ConstantFP::get(Ty, v);
  }
  if (Ty->isPointerTy()) {
    uint64_t raw = 0;
    std::memcpy(&raw, addr, sizeof(raw));
    return ConstantExpr::getIntToPtr(
        ConstantInt::get(Type::getInt64Ty(Ctx), raw), Ty);
  }
  return nullptr;
}

} // anonymous namespace

PreservedAnalyses
EJitStructFieldPass::run(Function &F, FunctionAnalysisManager &AM) {
  Module *M = F.getParent();
  if (!M)
    return PreservedAnalyses::all();

  const DataLayout &DL = M->getDataLayout();

  DenseMap<const GlobalVariable *, GVPeriodInfo> gvPeriodMap;
  buildGVPeriodMap(*M, gvPeriodMap);

  struct Replacement { LoadInst *LI; Constant *ConstVal; };
  SmallVector<Replacement, 16> replacements;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI || !LI->hasMetadata(MD_EJIT_MAY_CONST))
        continue;

      Value *PtrOp = LI->getPointerOperand();

      // Helper: resolve base address for a global variable
      auto resolveBase = [&](const GlobalVariable *GV,
                             const GVPeriodInfo &info) -> void * {
        if (info.isArray) {
          const auto *arrs = registry_.getArrays(info.periodName);
          if (!arrs || arrs->empty())
            return nullptr;
          // For single-array periods, use the only array's base.
          // For multi-array periods, look up by variable name to get the
          // correct array's base address (not just the first one).
          if (arrs->size() == 1)
            return arrs->front().baseAddr;
          const auto *paInfo = registry_.getArrayInfo(GV->getName().str());
          return paInfo ? paInfo->baseAddr : nullptr;
        }
        return registry_.getStaticVarAddr(GV->getName().str());
      };

      // Direct global variable load
      if (auto *GV = dyn_cast<GlobalVariable>(PtrOp->stripPointerCasts())) {
        auto it = gvPeriodMap.find(GV);
        if (it == gvPeriodMap.end())
          continue;

        void *base = resolveBase(GV, it->second);
        if (!base)
          continue;

        if (auto *C = createConstantFromMemory(base, LI->getType(), DL))
          replacements.push_back({LI, C});
        continue;
      }

      // GEP-based access
      if (auto *GEP = dyn_cast<GEPOperator>(PtrOp)) {
        const GlobalVariable *GV = findRootGV(GEP);
        if (!GV)
          continue;

        auto it = gvPeriodMap.find(GV);
        if (it == gvPeriodMap.end())
          continue;

        auto byteOffset = accumulateConstantOffset(DL, GEP);
        if (!byteOffset)
          continue;

        void *base = resolveBase(GV, it->second);
        if (!base)
          continue;

        uint8_t *fieldAddr = static_cast<uint8_t *>(base) + *byteOffset;
        if (auto *C = createConstantFromMemory(fieldAddr, LI->getType(), DL))
          replacements.push_back({LI, C});
      }
    }
  }

  bool changed = false;
  for (auto &R : replacements) {
    R.LI->replaceAllUsesWith(R.ConstVal);
    R.LI->eraseFromParent();
    changed = true;
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
