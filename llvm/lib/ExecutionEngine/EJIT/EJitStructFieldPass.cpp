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
#include "llvm/Support/Debug.h"
#include <cstring>

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-struct-field"

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

/// Walk a GEP chain from the load's pointer operand down to the root
/// global variable, accumulating the total byte offset. All GEP indices
/// must be constants (already folded by InstCombine after param substitution).
static std::optional<uint64_t>
accumulateFullOffset(const DataLayout &DL, const Value *PtrOp) {
  APInt total(DL.getPointerSizeInBits(0), 0);

  while (PtrOp) {
    PtrOp = PtrOp->stripPointerCasts();
    if (isa<GlobalVariable>(PtrOp))
      break;

    auto *GEP = dyn_cast<GEPOperator>(PtrOp);
    if (!GEP)
      return std::nullopt;

    // Compute the total offset for all indices of this GEP in one call.
    // getIndexedOffsetInType already returns the cumulative offset from the
    // start of the source element type through all given indices.
    SmallVector<Value *, 4> IdxList;
    bool allConstant = true;
    for (auto I = GEP->idx_begin(), E = GEP->idx_end(); I != E; ++I) {
      if (!isa<ConstantInt>(*I)) {
        allConstant = false;
        break;
      }
      IdxList.push_back(*I);
    }
    if (!allConstant)
      return std::nullopt;
    total += APInt(total.getBitWidth(),
                   DL.getIndexedOffsetInType(
                       GEP->getSourceElementType(), IdxList));

    PtrOp = GEP->getPointerOperand();
  }

  return total.getZExtValue();
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
      // On big-endian targets, memcpy places bytes at the MSB end of raw.
      // Shift them down so APInt sees the correct numeric value.
      if (!DL.isLittleEndian())
        raw >>= (8 - byteSize) * 8;
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

  // Build GV-level may_const field offset map (v1.7 fallback for when
  // optimization passes drop per-load !ejit.may_const metadata).
  DenseMap<const GlobalVariable *, SmallVector<uint64_t, 4>> mayConstFieldMap;
  for (GlobalVariable &GV : M->globals()) {
    MDNode *MD = GV.getMetadata(MD_EJIT_METADATA);
    if (!MD)
      continue;
    SmallVector<uint64_t, 4> offsets;
    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 2)
        continue;
      auto *Tag = dyn_cast<MDString>(Sub->getOperand(0));
      if (!Tag || Tag->getString() != TAG_EJIT_MAY_CONST_FIELD)
        continue;
      if (auto *CI = mdconst::dyn_extract<ConstantInt>(Sub->getOperand(1)))
        offsets.push_back(CI->getZExtValue());
    }
    if (!offsets.empty())
      mayConstFieldMap[&GV] = std::move(offsets);
  }

  struct Replacement { LoadInst *LI; Constant *ConstVal; };
  SmallVector<Replacement, 16> replacements;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        continue;
      bool isMayConst = LI->hasMetadata(MD_EJIT_MAY_CONST);

      if (!isMayConst) {
        // v1.7 fallback: check GV-level may_const field offsets
        Value *Ptr = LI->getPointerOperand();
        if (auto *RootGV = findRootGV(Ptr)) {
          auto It = mayConstFieldMap.find(RootGV);
          if (It != mayConstFieldMap.end()) {
            auto Off = accumulateFullOffset(DL, Ptr);
            if (Off && is_contained(It->second, *Off))
              isMayConst = true;
          }
        }
      }
      if (!isMayConst)
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
      // Walk the full GEP chain (may be multiple GEPs: array index + field)
      {
        const GlobalVariable *GV = findRootGV(PtrOp);
        if (GV) {
          auto it = gvPeriodMap.find(GV);
          if (it == gvPeriodMap.end())
            continue;

          auto byteOffset = accumulateFullOffset(DL, PtrOp);
          if (!byteOffset)
            continue;

          void *base = resolveBase(GV, it->second);
          if (!base)
            continue;

          uint8_t *fieldAddr = static_cast<uint8_t *>(base) + *byteOffset;
          if (auto *C = createConstantFromMemory(fieldAddr, LI->getType(), DL))
            replacements.push_back({LI, C});
          continue;
        }

        // Indirect path: load ptr from GV, then GEP into pointed-to data.
        // Pattern A: %ptr = load ptr, ptr @g_ptr  → GEP %S, ptr %ptr, ...
        // Pattern B: %ptr = load ptr, ptr @ptrArr[idx] → GEP %S, ptr %ptr, ...
        // Supported for pointer-type ejit_period and ejit_period_arr globals.
        const Value *V = PtrOp;
        SmallVector<const GEPOperator *, 4> FieldGEPs;
        while (V) {
          V = V->stripPointerCasts();
          if (auto *GEP = dyn_cast<GEPOperator>(V)) {
            FieldGEPs.push_back(GEP);
            V = GEP->getPointerOperand();
            continue;
          }
          break;
        }

        auto *BaseLoad = dyn_cast<LoadInst>(V);
        if (!BaseLoad)
          continue;

        // Resolve which GV the load reads from (direct or via ptr-array GEP).
        const Value *LoadPtr = BaseLoad->getPointerOperand()->stripPointerCasts();
        const GlobalVariable *PtrGV = nullptr;
        uint64_t ptrArrayByteOff = 0;

        if (auto *DirectGV = dyn_cast<GlobalVariable>(LoadPtr)) {
          PtrGV = DirectGV;
        } else if (auto *PtrGEP = dyn_cast<GEPOperator>(LoadPtr)) {
          PtrGV = dyn_cast<GlobalVariable>(
              PtrGEP->getPointerOperand()->stripPointerCasts());
          if (PtrGV) {
            SmallVector<Value *, 4> PtrIdxList;
            bool allConst = true;
            for (auto I = PtrGEP->idx_begin(), E = PtrGEP->idx_end();
                 I != E; ++I) {
              if (!isa<ConstantInt>(*I)) { allConst = false; break; }
              PtrIdxList.push_back(*I);
            }
            if (!allConst) continue;
            ptrArrayByteOff = DL.getIndexedOffsetInType(
                PtrGEP->getSourceElementType(), PtrIdxList);
          }
        }
        if (!PtrGV)
          continue;

        auto it = gvPeriodMap.find(PtrGV);
        if (it == gvPeriodMap.end())
          continue;

        void *gvBase = resolveBase(PtrGV, it->second);
        if (!gvBase)
          continue;

        // Read the pointer value stored in the GV: *(void**)(gvBase + ptrArrayByteOff)
        uintptr_t ptrSlot = reinterpret_cast<uintptr_t>(gvBase) + ptrArrayByteOff;
        void *dataBase = nullptr;
        std::memcpy(&dataBase, reinterpret_cast<void *>(ptrSlot), sizeof(void *));
        if (!dataBase)
          continue;

        // Compute field offset from GEPs past the pointer dereference.
        uint64_t fieldOff = 0;
        for (auto It = FieldGEPs.rbegin(); It != FieldGEPs.rend(); ++It) {
          SmallVector<Value *, 4> IdxList;
          for (auto I = (*It)->idx_begin(), E = (*It)->idx_end(); I != E; ++I) {
            if (!isa<ConstantInt>(*I)) { fieldOff = UINT64_MAX; break; }
            IdxList.push_back(*I);
          }
          if (fieldOff == UINT64_MAX) break;
          fieldOff += DL.getIndexedOffsetInType(
              (*It)->getSourceElementType(), IdxList);
        }
        if (fieldOff == UINT64_MAX)
          continue;

        uint8_t *fieldAddr = static_cast<uint8_t *>(dataBase) + fieldOff;
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

  LLVM_DEBUG(if (changed) dbgs() << "ejit-struct-field: replaced "
                                  << replacements.size() << " load(s) in "
                                  << F.getName() << "\n");
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
