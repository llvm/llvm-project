//===- WholeProgramDevirt.cpp - Whole program virtual call optimization ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements whole program optimization of virtual calls in cases
// where we know (via !type metadata) that the list of callees is fixed. This
// includes the following:
// - Single implementation devirtualization: if a virtual call has a single
//   possible callee, replace all calls with a direct call to that callee.
// - Virtual constant propagation: if the virtual function's return type is an
//   integer <=64 bits and all possible callees are readnone, for each class and
//   each list of constant arguments: evaluate the function, store the return
//   value alongside the virtual table, and rewrite each virtual call as a load
//   from the virtual table.
// - Uniform return value optimization: if the conditions for virtual constant
//   propagation hold and each function returns the same constant value, replace
//   each virtual call with that constant.
// - Unique return value optimization for i1 return values: if the conditions
//   for virtual constant propagation hold and a single vtable's function
//   returns 0, or a single vtable's function returns 1, replace each virtual
//   call with a comparison of the vptr against that vtable's address.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Evaluator.h"
#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>

using namespace llvm;
using namespace wholeprogramdevirt;

#define DEBUG_TYPE "wholeprogramdevirt"

// Find the minimum offset that we may store a value of size Size bits at. If
// IsAfter is set, look for an offset before the object, otherwise look for an
// offset after the object.
uint64_t
wholeprogramdevirt::findLowestOffset(ArrayRef<VirtualCallTarget> Targets,
                                     bool IsAfter, uint64_t Size) {
  // Find a minimum offset taking into account only vtable sizes.
  uint64_t MinByte = 0;
  for (const VirtualCallTarget &Target : Targets) {
    if (IsAfter)
      MinByte = std::max(MinByte, Target.minAfterBytes());
    else
      MinByte = std::max(MinByte, Target.minBeforeBytes());
  }

  // Build a vector of arrays of bytes covering, for each target, a slice of the
  // used region (see AccumBitVector::BytesUsed in
  // llvm/Transforms/IPO/WholeProgramDevirt.h) starting at MinByte. Effectively,
  // this aligns the used regions to start at MinByte.
  //
  // In this example, A, B and C are vtables, # is a byte already allocated for
  // a virtual function pointer, AAAA... (etc.) are the used regions for the
  // vtables and Offset(X) is the value computed for the Offset variable below
  // for X.
  //
  //                    Offset(A)
  //                    |       |
  //                            |MinByte
  // A: ################AAAAAAAA|AAAAAAAA
  // B: ########BBBBBBBBBBBBBBBB|BBBB
  // C: ########################|CCCCCCCCCCCCCCCC
  //            |   Offset(B)   |
  //
  // This code produces the slices of A, B and C that appear after the divider
  // at MinByte.
  std::vector<ArrayRef<uint8_t>> Used;
  for (const VirtualCallTarget &Target : Targets) {
    ArrayRef<uint8_t> VTUsed = IsAfter ? Target.TM->Bits->After.BytesUsed
                                       : Target.TM->Bits->Before.BytesUsed;
    uint64_t Offset = IsAfter ? MinByte - Target.minAfterBytes()
                              : MinByte - Target.minBeforeBytes();

    // Disregard used regions that are smaller than Offset. These are
    // effectively all-free regions that do not need to be checked.
    if (VTUsed.size() > Offset)
      Used.push_back(VTUsed.slice(Offset));
  }

  if (Size == 1) {
    // Find a free bit in each member of Used.
    for (unsigned I = 0;; ++I) {
      uint8_t BitsUsed = 0;
      for (auto &&B : Used)
        if (I < B.size())
          BitsUsed |= B[I];
      if (BitsUsed != 0xff)
        return (MinByte + I) * 8 +
               countTrailingZeros(uint8_t(~BitsUsed), ZB_Undefined);
    }
  } else {
    // Find a free (Size/8) byte region in each member of Used.
    // FIXME: see if alignment helps.
    for (unsigned I = 0;; ++I) {
      for (auto &&B : Used) {
        unsigned Byte = 0;
        while ((I + Byte) < B.size() && Byte < (Size / 8)) {
          if (B[I + Byte])
            goto NextI;
          ++Byte;
        }
      }
      return (MinByte + I) * 8;
    NextI:;
    }
  }
}

void wholeprogramdevirt::setBeforeReturnValues(
    MutableArrayRef<VirtualCallTarget> Targets, uint64_t AllocBefore,
    unsigned BitWidth, int64_t &OffsetByte, uint64_t &OffsetBit) {
  if (BitWidth == 1)
    OffsetByte = -(AllocBefore / 8 + 1);
  else
    OffsetByte = -((AllocBefore + 7) / 8 + (BitWidth + 7) / 8);
  OffsetBit = AllocBefore % 8;

  for (VirtualCallTarget &Target : Targets) {
    if (BitWidth == 1)
      Target.setBeforeBit(AllocBefore);
    else
      Target.setBeforeBytes(AllocBefore, (BitWidth + 7) / 8);
  }
}

void wholeprogramdevirt::setAfterReturnValues(
    MutableArrayRef<VirtualCallTarget> Targets, uint64_t AllocAfter,
    unsigned BitWidth, int64_t &OffsetByte, uint64_t &OffsetBit) {
  if (BitWidth == 1)
    OffsetByte = AllocAfter / 8;
  else
    OffsetByte = (AllocAfter + 7) / 8;
  OffsetBit = AllocAfter % 8;

  for (VirtualCallTarget &Target : Targets) {
    if (BitWidth == 1)
      Target.setAfterBit(AllocAfter);
    else
      Target.setAfterBytes(AllocAfter, (BitWidth + 7) / 8);
  }
}

VirtualCallTarget::VirtualCallTarget(Function *Fn, const TypeMemberInfo *TM)
    : Fn(Fn), TM(TM),
      IsBigEndian(Fn->getParent()->getDataLayout().isBigEndian()), WasDevirt(false) {}

namespace {

// A slot in a set of virtual tables. The TypeID identifies the set of virtual
// tables, and the ByteOffset is the offset in bytes from the address point to
// the virtual function pointer.
struct VTableSlot {
  Metadata *TypeID;
  uint64_t ByteOffset;
};

} // end anonymous namespace

namespace llvm {

template <> struct DenseMapInfo<VTableSlot> {
  static VTableSlot getEmptyKey() {
    return {DenseMapInfo<Metadata *>::getEmptyKey(),
            DenseMapInfo<uint64_t>::getEmptyKey()};
  }
  static VTableSlot getTombstoneKey() {
    return {DenseMapInfo<Metadata *>::getTombstoneKey(),
            DenseMapInfo<uint64_t>::getTombstoneKey()};
  }
  static unsigned getHashValue(const VTableSlot &I) {
    return DenseMapInfo<Metadata *>::getHashValue(I.TypeID) ^
           DenseMapInfo<uint64_t>::getHashValue(I.ByteOffset);
  }
  static bool isEqual(const VTableSlot &LHS,
                      const VTableSlot &RHS) {
    return LHS.TypeID == RHS.TypeID && LHS.ByteOffset == RHS.ByteOffset;
  }
};

} // end namespace llvm

namespace {

// A virtual call site. VTable is the loaded virtual table pointer, and CS is
// the indirect virtual call.
struct VirtualCallSite {
  Value *VTable;
  CallSite CS;

  // If non-null, this field points to the associated unsafe use count stored in
  // the DevirtModule::NumUnsafeUsesForTypeTest map below. See the description
  // of that field for details.
  unsigned *NumUnsafeUses;

  void emitRemark(const Twine &OptName, const Twine &TargetName) {
    Function *F = CS.getCaller();
    emitOptimizationRemark(
        F->getContext(), DEBUG_TYPE, *F,
        CS.getInstruction()->getDebugLoc(),
        OptName + ": devirtualized a call to " + TargetName);
  }

  void replaceAndErase(const Twine &OptName, const Twine &TargetName,
                       bool RemarksEnabled, Value *New) {
    if (RemarksEnabled)
      emitRemark(OptName, TargetName);
    CS->replaceAllUsesWith(New);
    if (auto II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      BranchInst::Create(II->getNormalDest(), CS.getInstruction());
      II->getUnwindDest()->removePredecessor(II->getParent());
    }
    CS->eraseFromParent();
    // This use is no longer unsafe.
    if (NumUnsafeUses)
      --*NumUnsafeUses;
  }
};

struct DevirtModule {
  Module &M;
  IntegerType *Int8Ty;
  PointerType *Int8PtrTy;
  IntegerType *Int32Ty;

  bool RemarksEnabled;

  MapVector<VTableSlot, std::vector<VirtualCallSite>> CallSlots;

  // This map keeps track of the number of "unsafe" uses of a loaded function
  // pointer. The key is the associated llvm.type.test intrinsic call generated
  // by this pass. An unsafe use is one that calls the loaded function pointer
  // directly. Every time we eliminate an unsafe use (for example, by
  // devirtualizing it or by applying virtual constant propagation), we
  // decrement the value stored in this map. If a value reaches zero, we can
  // eliminate the type check by RAUWing the associated llvm.type.test call with
  // true.
  std::map<CallInst *, unsigned> NumUnsafeUsesForTypeTest;

  DevirtModule(Module &M)
      : M(M), Int8Ty(Type::getInt8Ty(M.getContext())),
        Int8PtrTy(Type::getInt8PtrTy(M.getContext())),
        Int32Ty(Type::getInt32Ty(M.getContext())),
        RemarksEnabled(areRemarksEnabled()) {}

  bool areRemarksEnabled();

  void scanTypeTestUsers(Function *TypeTestFunc, Function *AssumeFunc);
  void scanTypeCheckedLoadUsers(Function *TypeCheckedLoadFunc);

  void buildTypeIdentifierMap(
      std::vector<VTableBits> &Bits,
      DenseMap<Metadata *, std::set<TypeMemberInfo>> &TypeIdMap);
  bool
  tryFindVirtualCallTargets(std::vector<VirtualCallTarget> &TargetsForSlot,
                            const std::set<TypeMemberInfo> &TypeMemberInfos,
                            uint64_t ByteOffset);
  bool trySingleImplDevirt(MutableArrayRef<VirtualCallTarget> TargetsForSlot,
                           MutableArrayRef<VirtualCallSite> CallSites);
  bool tryEvaluateFunctionsWithArgs(
      MutableArrayRef<VirtualCallTarget> TargetsForSlot,
      ArrayRef<ConstantInt *> Args);
  bool tryUniformRetValOpt(IntegerType *RetType,
                           MutableArrayRef<VirtualCallTarget> TargetsForSlot,
                           MutableArrayRef<VirtualCallSite> CallSites);
  bool tryUniqueRetValOpt(unsigned BitWidth,
                          MutableArrayRef<VirtualCallTarget> TargetsForSlot,
                          MutableArrayRef<VirtualCallSite> CallSites);
  bool tryVirtualConstProp(MutableArrayRef<VirtualCallTarget> TargetsForSlot,
                           ArrayRef<VirtualCallSite> CallSites);

  void rebuildGlobal(VTableBits &B);

  bool run();
};

struct WholeProgramDevirt : public ModulePass {
  static char ID;

  WholeProgramDevirt() : ModulePass(ID) {
    initializeWholeProgramDevirtPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    return DevirtModule(M).run();
  }
};

} // end anonymous namespace

INITIALIZE_PASS(WholeProgramDevirt, "wholeprogramdevirt",
                "Whole program devirtualization", false, false)
char WholeProgramDevirt::ID = 0;

ModulePass *llvm::createWholeProgramDevirtPass() {
  return new WholeProgramDevirt;
}

PreservedAnalyses WholeProgramDevirtPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  if (!DevirtModule(M).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

void DevirtModule::buildTypeIdentifierMap(
    std::vector<VTableBits> &Bits,
    DenseMap<Metadata *, std::set<TypeMemberInfo>> &TypeIdMap) {
  DenseMap<GlobalVariable *, VTableBits *> GVToBits;
  Bits.reserve(M.getGlobalList().size());
  SmallVector<MDNode *, 2> Types;
  for (GlobalVariable &GV : M.globals()) {
    Types.clear();
    GV.getMetadata(LLVMContext::MD_type, Types);
    if (Types.empty())
      continue;

    VTableBits *&BitsPtr = GVToBits[&GV];
    if (!BitsPtr) {
      Bits.emplace_back();
      Bits.back().GV = &GV;
      Bits.back().ObjectSize =
          M.getDataLayout().getTypeAllocSize(GV.getInitializer()->getType());
      BitsPtr = &Bits.back();
    }

    for (MDNode *Type : Types) {
      auto TypeID = Type->getOperand(1).get();

      uint64_t Offset =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
              ->getZExtValue();

      TypeIdMap[TypeID].insert({BitsPtr, Offset});
    }
  }
}

bool DevirtModule::tryFindVirtualCallTargets(
    std::vector<VirtualCallTarget> &TargetsForSlot,
    const std::set<TypeMemberInfo> &TypeMemberInfos, uint64_t ByteOffset) {
  for (const TypeMemberInfo &TM : TypeMemberInfos) {
    if (!TM.Bits->GV->isConstant())
      return false;

    auto Init = dyn_cast<ConstantArray>(TM.Bits->GV->getInitializer());
    if (!Init)
      return false;
    ArrayType *VTableTy = Init->getType();

    uint64_t ElemSize =
        M.getDataLayout().getTypeAllocSize(VTableTy->getElementType());
    uint64_t GlobalSlotOffset = TM.Offset + ByteOffset;
    if (GlobalSlotOffset % ElemSize != 0)
      return false;

    unsigned Op = GlobalSlotOffset / ElemSize;
    if (Op >= Init->getNumOperands())
      return false;

    auto Fn = dyn_cast<Function>(Init->getOperand(Op)->stripPointerCasts());
    if (!Fn)
      return false;

    // We can disregard __cxa_pure_virtual as a possible call target, as
    // calls to pure virtuals are UB.
    if (Fn->getName() == "__cxa_pure_virtual")
      continue;

    TargetsForSlot.push_back({Fn, &TM});
  }

  // Give up if we couldn't find any targets.
  return !TargetsForSlot.empty();
}

bool DevirtModule::trySingleImplDevirt(
    MutableArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // See if the program contains a single implementation of this virtual
  // function.
  Function *TheFn = TargetsForSlot[0].Fn;
  for (auto &&Target : TargetsForSlot)
    if (TheFn != Target.Fn)
      return false;

  if (RemarksEnabled)
    TargetsForSlot[0].WasDevirt = true;
  // If so, update each call site to call that implementation directly.
  for (auto &&VCallSite : CallSites) {
    if (RemarksEnabled)
      VCallSite.emitRemark("single-impl", TheFn->getName());
    VCallSite.CS.setCalledFunction(ConstantExpr::getBitCast(
        TheFn, VCallSite.CS.getCalledValue()->getType()));
    // This use is no longer unsafe.
    if (VCallSite.NumUnsafeUses)
      --*VCallSite.NumUnsafeUses;
  }
  return true;
}

bool DevirtModule::tryEvaluateFunctionsWithArgs(
    MutableArrayRef<VirtualCallTarget> TargetsForSlot,
    ArrayRef<ConstantInt *> Args) {
  // Evaluate each function and store the result in each target's RetVal
  // field.
  for (VirtualCallTarget &Target : TargetsForSlot) {
    if (Target.Fn->arg_size() != Args.size() + 1)
      return false;
    for (unsigned I = 0; I != Args.size(); ++I)
      if (Target.Fn->getFunctionType()->getParamType(I + 1) !=
          Args[I]->getType())
        return false;

    Evaluator Eval(M.getDataLayout(), nullptr);
    SmallVector<Constant *, 2> EvalArgs;
    EvalArgs.push_back(
        Constant::getNullValue(Target.Fn->getFunctionType()->getParamType(0)));
    EvalArgs.insert(EvalArgs.end(), Args.begin(), Args.end());
    Constant *RetVal;
    if (!Eval.EvaluateFunction(Target.Fn, RetVal, EvalArgs) ||
        !isa<ConstantInt>(RetVal))
      return false;
    Target.RetVal = cast<ConstantInt>(RetVal)->getZExtValue();
  }
  return true;
}

bool DevirtModule::tryUniformRetValOpt(
    IntegerType *RetType, MutableArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // Uniform return value optimization. If all functions return the same
  // constant, replace all calls with that constant.
  uint64_t TheRetVal = TargetsForSlot[0].RetVal;
  for (const VirtualCallTarget &Target : TargetsForSlot)
    if (Target.RetVal != TheRetVal)
      return false;

  auto TheRetValConst = ConstantInt::get(RetType, TheRetVal);
  for (auto Call : CallSites)
    Call.replaceAndErase("uniform-ret-val", TargetsForSlot[0].Fn->getName(),
                         RemarksEnabled, TheRetValConst);
  if (RemarksEnabled)
    for (auto &&Target : TargetsForSlot)
      Target.WasDevirt = true;
  return true;
}

bool DevirtModule::tryUniqueRetValOpt(
    unsigned BitWidth, MutableArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // IsOne controls whether we look for a 0 or a 1.
  auto tryUniqueRetValOptFor = [&](bool IsOne) {
    const TypeMemberInfo *UniqueMember = nullptr;
    for (const VirtualCallTarget &Target : TargetsForSlot) {
      if (Target.RetVal == (IsOne ? 1 : 0)) {
        if (UniqueMember)
          return false;
        UniqueMember = Target.TM;
      }
    }

    // We should have found a unique member or bailed out by now. We already
    // checked for a uniform return value in tryUniformRetValOpt.
    assert(UniqueMember);

    // Replace each call with the comparison.
    for (auto &&Call : CallSites) {
      IRBuilder<> B(Call.CS.getInstruction());
      Value *OneAddr = B.CreateBitCast(UniqueMember->Bits->GV, Int8PtrTy);
      OneAddr = B.CreateConstGEP1_64(OneAddr, UniqueMember->Offset);
      Value *Cmp = B.CreateICmp(IsOne ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE,
                                Call.VTable, OneAddr);
      Call.replaceAndErase("unique-ret-val", TargetsForSlot[0].Fn->getName(),
                           RemarksEnabled, Cmp);
    }
    // Update devirtualization statistics for targets.
    if (RemarksEnabled)
      for (auto &&Target : TargetsForSlot)
        Target.WasDevirt = true;

    return true;
  };

  if (BitWidth == 1) {
    if (tryUniqueRetValOptFor(true))
      return true;
    if (tryUniqueRetValOptFor(false))
      return true;
  }
  return false;
}

bool DevirtModule::tryVirtualConstProp(
    MutableArrayRef<VirtualCallTarget> TargetsForSlot,
    ArrayRef<VirtualCallSite> CallSites) {
  // This only works if the function returns an integer.
  auto RetType = dyn_cast<IntegerType>(TargetsForSlot[0].Fn->getReturnType());
  if (!RetType)
    return false;
  unsigned BitWidth = RetType->getBitWidth();
  if (BitWidth > 64)
    return false;

  // Make sure that each function does not access memory, takes at least one
  // argument, does not use its first argument (which we assume is 'this'),
  // and has the same return type.
  for (VirtualCallTarget &Target : TargetsForSlot) {
    if (!Target.Fn->doesNotAccessMemory() || Target.Fn->arg_empty() ||
        !Target.Fn->arg_begin()->use_empty() ||
        Target.Fn->getReturnType() != RetType)
      return false;
  }

  // Group call sites by the list of constant arguments they pass.
  // The comparator ensures deterministic ordering.
  struct ByAPIntValue {
    bool operator()(const std::vector<ConstantInt *> &A,
                    const std::vector<ConstantInt *> &B) const {
      return std::lexicographical_compare(
          A.begin(), A.end(), B.begin(), B.end(),
          [](ConstantInt *AI, ConstantInt *BI) {
            return AI->getValue().ult(BI->getValue());
          });
    }
  };
  std::map<std::vector<ConstantInt *>, std::vector<VirtualCallSite>,
           ByAPIntValue>
      VCallSitesByConstantArg;
  for (auto &&VCallSite : CallSites) {
    std::vector<ConstantInt *> Args;
    if (VCallSite.CS.getType() != RetType)
      continue;
    for (auto &&Arg :
         make_range(VCallSite.CS.arg_begin() + 1, VCallSite.CS.arg_end())) {
      if (!isa<ConstantInt>(Arg))
        break;
      Args.push_back(cast<ConstantInt>(&Arg));
    }
    if (Args.size() + 1 != VCallSite.CS.arg_size())
      continue;

    VCallSitesByConstantArg[Args].push_back(VCallSite);
  }

  for (auto &&CSByConstantArg : VCallSitesByConstantArg) {
    if (!tryEvaluateFunctionsWithArgs(TargetsForSlot, CSByConstantArg.first))
      continue;

    if (tryUniformRetValOpt(RetType, TargetsForSlot, CSByConstantArg.second))
      continue;

    if (tryUniqueRetValOpt(BitWidth, TargetsForSlot, CSByConstantArg.second))
      continue;

    // Find an allocation offset in bits in all vtables associated with the
    // type.
    uint64_t AllocBefore =
        findLowestOffset(TargetsForSlot, /*IsAfter=*/false, BitWidth);
    uint64_t AllocAfter =
        findLowestOffset(TargetsForSlot, /*IsAfter=*/true, BitWidth);

    // Calculate the total amount of padding needed to store a value at both
    // ends of the object.
    uint64_t TotalPaddingBefore = 0, TotalPaddingAfter = 0;
    for (auto &&Target : TargetsForSlot) {
      TotalPaddingBefore += std::max<int64_t>(
          (AllocBefore + 7) / 8 - Target.allocatedBeforeBytes() - 1, 0);
      TotalPaddingAfter += std::max<int64_t>(
          (AllocAfter + 7) / 8 - Target.allocatedAfterBytes() - 1, 0);
    }

    // If the amount of padding is too large, give up.
    // FIXME: do something smarter here.
    if (std::min(TotalPaddingBefore, TotalPaddingAfter) > 128)
      continue;

    // Calculate the offset to the value as a (possibly negative) byte offset
    // and (if applicable) a bit offset, and store the values in the targets.
    int64_t OffsetByte;
    uint64_t OffsetBit;
    if (TotalPaddingBefore <= TotalPaddingAfter)
      setBeforeReturnValues(TargetsForSlot, AllocBefore, BitWidth, OffsetByte,
                            OffsetBit);
    else
      setAfterReturnValues(TargetsForSlot, AllocAfter, BitWidth, OffsetByte,
                           OffsetBit);

    if (RemarksEnabled)
      for (auto &&Target : TargetsForSlot)
        Target.WasDevirt = true;

    // Rewrite each call to a load from OffsetByte/OffsetBit.
    for (auto Call : CSByConstantArg.second) {
      IRBuilder<> B(Call.CS.getInstruction());
      Value *Addr = B.CreateConstGEP1_64(Call.VTable, OffsetByte);
      if (BitWidth == 1) {
        Value *Bits = B.CreateLoad(Addr);
        Value *Bit = ConstantInt::get(Int8Ty, 1ULL << OffsetBit);
        Value *BitsAndBit = B.CreateAnd(Bits, Bit);
        auto IsBitSet = B.CreateICmpNE(BitsAndBit, ConstantInt::get(Int8Ty, 0));
        Call.replaceAndErase("virtual-const-prop-1-bit",
                             TargetsForSlot[0].Fn->getName(),
                             RemarksEnabled, IsBitSet);
      } else {
        Value *ValAddr = B.CreateBitCast(Addr, RetType->getPointerTo());
        Value *Val = B.CreateLoad(RetType, ValAddr);
        Call.replaceAndErase("virtual-const-prop",
                             TargetsForSlot[0].Fn->getName(),
                             RemarksEnabled, Val);
      }
    }
  }
  return true;
}

void DevirtModule::rebuildGlobal(VTableBits &B) {
  if (B.Before.Bytes.empty() && B.After.Bytes.empty())
    return;

  // Align each byte array to pointer width.
  unsigned PointerSize = M.getDataLayout().getPointerSize();
  B.Before.Bytes.resize(alignTo(B.Before.Bytes.size(), PointerSize));
  B.After.Bytes.resize(alignTo(B.After.Bytes.size(), PointerSize));

  // Before was stored in reverse order; flip it now.
  for (size_t I = 0, Size = B.Before.Bytes.size(); I != Size / 2; ++I)
    std::swap(B.Before.Bytes[I], B.Before.Bytes[Size - 1 - I]);

  // Build an anonymous global containing the before bytes, followed by the
  // original initializer, followed by the after bytes.
  auto NewInit = ConstantStruct::getAnon(
      {ConstantDataArray::get(M.getContext(), B.Before.Bytes),
       B.GV->getInitializer(),
       ConstantDataArray::get(M.getContext(), B.After.Bytes)});
  auto NewGV =
      new GlobalVariable(M, NewInit->getType(), B.GV->isConstant(),
                         GlobalVariable::PrivateLinkage, NewInit, "", B.GV);
  NewGV->setSection(B.GV->getSection());
  NewGV->setComdat(B.GV->getComdat());

  // Copy the original vtable's metadata to the anonymous global, adjusting
  // offsets as required.
  NewGV->copyMetadata(B.GV, B.Before.Bytes.size());

  // Build an alias named after the original global, pointing at the second
  // element (the original initializer).
  auto Alias = GlobalAlias::create(
      B.GV->getInitializer()->getType(), 0, B.GV->getLinkage(), "",
      ConstantExpr::getGetElementPtr(
          NewInit->getType(), NewGV,
          ArrayRef<Constant *>{ConstantInt::get(Int32Ty, 0),
                               ConstantInt::get(Int32Ty, 1)}),
      &M);
  Alias->setVisibility(B.GV->getVisibility());
  Alias->takeName(B.GV);

  B.GV->replaceAllUsesWith(Alias);
  B.GV->eraseFromParent();
}

bool DevirtModule::areRemarksEnabled() {
  const auto &FL = M.getFunctionList();
  if (FL.empty())
    return false;
  const Function &Fn = FL.front();
  auto DI = DiagnosticInfoOptimizationRemark(DEBUG_TYPE, Fn, DebugLoc(), "");
  return DI.isEnabled();
}

void DevirtModule::scanTypeTestUsers(Function *TypeTestFunc,
                                     Function *AssumeFunc) {
  // Find all virtual calls via a virtual table pointer %p under an assumption
  // of the form llvm.assume(llvm.type.test(%p, %md)). This indicates that %p
  // points to a member of the type identifier %md. Group calls by (type ID,
  // offset) pair (effectively the identity of the virtual function) and store
  // to CallSlots.
  DenseSet<Value *> SeenPtrs;
  for (auto I = TypeTestFunc->use_begin(), E = TypeTestFunc->use_end();
       I != E;) {
    auto CI = dyn_cast<CallInst>(I->getUser());
    ++I;
    if (!CI)
      continue;

    // Search for virtual calls based on %p and add them to DevirtCalls.
    SmallVector<DevirtCallSite, 1> DevirtCalls;
    SmallVector<CallInst *, 1> Assumes;
    findDevirtualizableCallsForTypeTest(DevirtCalls, Assumes, CI);

    // If we found any, add them to CallSlots. Only do this if we haven't seen
    // the vtable pointer before, as it may have been CSE'd with pointers from
    // other call sites, and we don't want to process call sites multiple times.
    if (!Assumes.empty()) {
      Metadata *TypeId =
          cast<MetadataAsValue>(CI->getArgOperand(1))->getMetadata();
      Value *Ptr = CI->getArgOperand(0)->stripPointerCasts();
      if (SeenPtrs.insert(Ptr).second) {
        for (DevirtCallSite Call : DevirtCalls) {
          CallSlots[{TypeId, Call.Offset}].push_back(
              {CI->getArgOperand(0), Call.CS, nullptr});
        }
      }
    }

    // We no longer need the assumes or the type test.
    for (auto Assume : Assumes)
      Assume->eraseFromParent();
    // We can't use RecursivelyDeleteTriviallyDeadInstructions here because we
    // may use the vtable argument later.
    if (CI->use_empty())
      CI->eraseFromParent();
  }
}

void DevirtModule::scanTypeCheckedLoadUsers(Function *TypeCheckedLoadFunc) {
  Function *TypeTestFunc = Intrinsic::getDeclaration(&M, Intrinsic::type_test);

  for (auto I = TypeCheckedLoadFunc->use_begin(),
            E = TypeCheckedLoadFunc->use_end();
       I != E;) {
    auto CI = dyn_cast<CallInst>(I->getUser());
    ++I;
    if (!CI)
      continue;

    Value *Ptr = CI->getArgOperand(0);
    Value *Offset = CI->getArgOperand(1);
    Value *TypeIdValue = CI->getArgOperand(2);
    Metadata *TypeId = cast<MetadataAsValue>(TypeIdValue)->getMetadata();

    SmallVector<DevirtCallSite, 1> DevirtCalls;
    SmallVector<Instruction *, 1> LoadedPtrs;
    SmallVector<Instruction *, 1> Preds;
    bool HasNonCallUses = false;
    findDevirtualizableCallsForTypeCheckedLoad(DevirtCalls, LoadedPtrs, Preds,
                                               HasNonCallUses, CI);

    // Start by generating "pessimistic" code that explicitly loads the function
    // pointer from the vtable and performs the type check. If possible, we will
    // eliminate the load and the type check later.

    // If possible, only generate the load at the point where it is used.
    // This helps avoid unnecessary spills.
    IRBuilder<> LoadB(
        (LoadedPtrs.size() == 1 && !HasNonCallUses) ? LoadedPtrs[0] : CI);
    Value *GEP = LoadB.CreateGEP(Int8Ty, Ptr, Offset);
    Value *GEPPtr = LoadB.CreateBitCast(GEP, PointerType::getUnqual(Int8PtrTy));
    Value *LoadedValue = LoadB.CreateLoad(Int8PtrTy, GEPPtr);

    for (Instruction *LoadedPtr : LoadedPtrs) {
      LoadedPtr->replaceAllUsesWith(LoadedValue);
      LoadedPtr->eraseFromParent();
    }

    // Likewise for the type test.
    IRBuilder<> CallB((Preds.size() == 1 && !HasNonCallUses) ? Preds[0] : CI);
    CallInst *TypeTestCall = CallB.CreateCall(TypeTestFunc, {Ptr, TypeIdValue});

    for (Instruction *Pred : Preds) {
      Pred->replaceAllUsesWith(TypeTestCall);
      Pred->eraseFromParent();
    }

    // We have already erased any extractvalue instructions that refer to the
    // intrinsic call, but the intrinsic may have other non-extractvalue uses
    // (although this is unlikely). In that case, explicitly build a pair and
    // RAUW it.
    if (!CI->use_empty()) {
      Value *Pair = UndefValue::get(CI->getType());
      IRBuilder<> B(CI);
      Pair = B.CreateInsertValue(Pair, LoadedValue, {0});
      Pair = B.CreateInsertValue(Pair, TypeTestCall, {1});
      CI->replaceAllUsesWith(Pair);
    }

    // The number of unsafe uses is initially the number of uses.
    auto &NumUnsafeUses = NumUnsafeUsesForTypeTest[TypeTestCall];
    NumUnsafeUses = DevirtCalls.size();

    // If the function pointer has a non-call user, we cannot eliminate the type
    // check, as one of those users may eventually call the pointer. Increment
    // the unsafe use count to make sure it cannot reach zero.
    if (HasNonCallUses)
      ++NumUnsafeUses;
    for (DevirtCallSite Call : DevirtCalls) {
      CallSlots[{TypeId, Call.Offset}].push_back(
          {Ptr, Call.CS, &NumUnsafeUses});
    }

    CI->eraseFromParent();
  }
}

bool DevirtModule::run() {
  Function *TypeTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_test));
  Function *TypeCheckedLoadFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_checked_load));
  Function *AssumeFunc = M.getFunction(Intrinsic::getName(Intrinsic::assume));

  if ((!TypeTestFunc || TypeTestFunc->use_empty() || !AssumeFunc ||
       AssumeFunc->use_empty()) &&
      (!TypeCheckedLoadFunc || TypeCheckedLoadFunc->use_empty()))
    return false;

  if (TypeTestFunc && AssumeFunc)
    scanTypeTestUsers(TypeTestFunc, AssumeFunc);

  if (TypeCheckedLoadFunc)
    scanTypeCheckedLoadUsers(TypeCheckedLoadFunc);

  // Rebuild type metadata into a map for easy lookup.
  std::vector<VTableBits> Bits;
  DenseMap<Metadata *, std::set<TypeMemberInfo>> TypeIdMap;
  buildTypeIdentifierMap(Bits, TypeIdMap);
  if (TypeIdMap.empty())
    return true;

  // For each (type, offset) pair:
  bool DidVirtualConstProp = false;
  std::map<std::string, Function*> DevirtTargets;
  for (auto &S : CallSlots) {
    // Search each of the members of the type identifier for the virtual
    // function implementation at offset S.first.ByteOffset, and add to
    // TargetsForSlot.
    std::vector<VirtualCallTarget> TargetsForSlot;
    if (!tryFindVirtualCallTargets(TargetsForSlot, TypeIdMap[S.first.TypeID],
                                   S.first.ByteOffset))
      continue;

    if (!trySingleImplDevirt(TargetsForSlot, S.second) &&
        tryVirtualConstProp(TargetsForSlot, S.second))
        DidVirtualConstProp = true;

    // Collect functions devirtualized at least for one call site for stats.
    if (RemarksEnabled)
      for (const auto &T : TargetsForSlot)
        if (T.WasDevirt)
          DevirtTargets[T.Fn->getName()] = T.Fn;
  }

  if (RemarksEnabled) {
    // Generate remarks for each devirtualized function.
    for (const auto &DT : DevirtTargets) {
      Function *F = DT.second;
      DISubprogram *SP = F->getSubprogram();
      DebugLoc DL = SP ? DebugLoc::get(SP->getScopeLine(), 0, SP) : DebugLoc();
      emitOptimizationRemark(F->getContext(), DEBUG_TYPE, *F, DL,
                             Twine("devirtualized ") + F->getName());
    }
  }

  // If we were able to eliminate all unsafe uses for a type checked load,
  // eliminate the type test by replacing it with true.
  if (TypeCheckedLoadFunc) {
    auto True = ConstantInt::getTrue(M.getContext());
    for (auto &&U : NumUnsafeUsesForTypeTest) {
      if (U.second == 0) {
        U.first->replaceAllUsesWith(True);
        U.first->eraseFromParent();
      }
    }
  }

  // Rebuild each global we touched as part of virtual constant propagation to
  // include the before and after bytes.
  if (DidVirtualConstProp)
    for (VTableBits &B : Bits)
      rebuildGlobal(B);

  return true;
}
