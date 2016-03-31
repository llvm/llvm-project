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
// where we know (via bitset information) that the list of callee is fixed. This
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
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/Transforms/Utils/Local.h"

#include <set>

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
    ArrayRef<uint8_t> VTUsed = IsAfter ? Target.BS->Bits->After.BytesUsed
                                       : Target.BS->Bits->Before.BytesUsed;
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

VirtualCallTarget::VirtualCallTarget(Function *Fn, const BitSetInfo *BS)
    : Fn(Fn), BS(BS),
      IsBigEndian(Fn->getParent()->getDataLayout().isBigEndian()) {}

namespace {

// A slot in a set of virtual tables. The BitSetID identifies the set of virtual
// tables, and the ByteOffset is the offset in bytes from the address point to
// the virtual function pointer.
struct VTableSlot {
  Metadata *BitSetID;
  uint64_t ByteOffset;
};

}

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
    return DenseMapInfo<Metadata *>::getHashValue(I.BitSetID) ^
           DenseMapInfo<uint64_t>::getHashValue(I.ByteOffset);
  }
  static bool isEqual(const VTableSlot &LHS,
                      const VTableSlot &RHS) {
    return LHS.BitSetID == RHS.BitSetID && LHS.ByteOffset == RHS.ByteOffset;
  }
};

}

namespace {

// A virtual call site. VTable is the loaded virtual table pointer, and CS is
// the indirect virtual call.
struct VirtualCallSite {
  Value *VTable;
  CallSite CS;

  void replaceAndErase(Value *New) {
    CS->replaceAllUsesWith(New);
    if (auto II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      BranchInst::Create(II->getNormalDest(), CS.getInstruction());
      II->getUnwindDest()->removePredecessor(II->getParent());
    }
    CS->eraseFromParent();
  }
};

struct DevirtModule {
  Module &M;
  IntegerType *Int8Ty;
  PointerType *Int8PtrTy;
  IntegerType *Int32Ty;

  MapVector<VTableSlot, std::vector<VirtualCallSite>> CallSlots;

  DevirtModule(Module &M)
      : M(M), Int8Ty(Type::getInt8Ty(M.getContext())),
        Int8PtrTy(Type::getInt8PtrTy(M.getContext())),
        Int32Ty(Type::getInt32Ty(M.getContext())) {}
  void findLoadCallsAtConstantOffset(Metadata *BitSet, Value *Ptr,
                                     uint64_t Offset, Value *VTable);
  void findCallsAtConstantOffset(Metadata *BitSet, Value *Ptr, uint64_t Offset,
                                 Value *VTable);

  void buildBitSets(std::vector<VTableBits> &Bits,
                    DenseMap<Metadata *, std::set<BitSetInfo>> &BitSets);
  bool tryFindVirtualCallTargets(std::vector<VirtualCallTarget> &TargetsForSlot,
                                 const std::set<BitSetInfo> &BitSetInfos,
                                 uint64_t ByteOffset);
  bool trySingleImplDevirt(ArrayRef<VirtualCallTarget> TargetsForSlot,
                           MutableArrayRef<VirtualCallSite> CallSites);
  bool tryEvaluateFunctionsWithArgs(
      MutableArrayRef<VirtualCallTarget> TargetsForSlot,
      ArrayRef<ConstantInt *> Args);
  bool tryUniformRetValOpt(IntegerType *RetType,
                           ArrayRef<VirtualCallTarget> TargetsForSlot,
                           MutableArrayRef<VirtualCallSite> CallSites);
  bool tryUniqueRetValOpt(unsigned BitWidth,
                          ArrayRef<VirtualCallTarget> TargetsForSlot,
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
  bool runOnModule(Module &M) { return DevirtModule(M).run(); }
};

} // anonymous namespace

INITIALIZE_PASS(WholeProgramDevirt, "wholeprogramdevirt",
                "Whole program devirtualization", false, false)
char WholeProgramDevirt::ID = 0;

ModulePass *llvm::createWholeProgramDevirtPass() {
  return new WholeProgramDevirt;
}

// Search for virtual calls that call FPtr and add them to CallSlots.
void DevirtModule::findCallsAtConstantOffset(Metadata *BitSet, Value *FPtr,
                                             uint64_t Offset, Value *VTable) {
  for (const Use &U : FPtr->uses()) {
    Value *User = U.getUser();
    if (isa<BitCastInst>(User)) {
      findCallsAtConstantOffset(BitSet, User, Offset, VTable);
    } else if (auto CI = dyn_cast<CallInst>(User)) {
      CallSlots[{BitSet, Offset}].push_back({VTable, CI});
    } else if (auto II = dyn_cast<InvokeInst>(User)) {
      CallSlots[{BitSet, Offset}].push_back({VTable, II});
    }
  }
}

// Search for virtual calls that load from VPtr and add them to CallSlots.
void DevirtModule::findLoadCallsAtConstantOffset(Metadata *BitSet, Value *VPtr,
                                                 uint64_t Offset,
                                                 Value *VTable) {
  for (const Use &U : VPtr->uses()) {
    Value *User = U.getUser();
    if (isa<BitCastInst>(User)) {
      findLoadCallsAtConstantOffset(BitSet, User, Offset, VTable);
    } else if (isa<LoadInst>(User)) {
      findCallsAtConstantOffset(BitSet, User, Offset, VTable);
    } else if (auto GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Take into account the GEP offset.
      if (VPtr == GEP->getPointerOperand() && GEP->hasAllConstantIndices()) {
        SmallVector<Value *, 8> Indices(GEP->op_begin() + 1, GEP->op_end());
        uint64_t GEPOffset = M.getDataLayout().getIndexedOffsetInType(
            GEP->getSourceElementType(), Indices);
        findLoadCallsAtConstantOffset(BitSet, User, Offset + GEPOffset, VTable);
      }
    }
  }
}

void DevirtModule::buildBitSets(
    std::vector<VTableBits> &Bits,
    DenseMap<Metadata *, std::set<BitSetInfo>> &BitSets) {
  NamedMDNode *BitSetNM = M.getNamedMetadata("llvm.bitsets");
  if (!BitSetNM)
    return;

  DenseMap<GlobalVariable *, VTableBits *> GVToBits;
  Bits.reserve(BitSetNM->getNumOperands());
  for (auto Op : BitSetNM->operands()) {
    auto OpConstMD = dyn_cast_or_null<ConstantAsMetadata>(Op->getOperand(1));
    if (!OpConstMD)
      continue;
    auto BitSetID = Op->getOperand(0).get();

    Constant *OpConst = OpConstMD->getValue();
    if (auto GA = dyn_cast<GlobalAlias>(OpConst))
      OpConst = GA->getAliasee();
    auto OpGlobal = dyn_cast<GlobalVariable>(OpConst);
    if (!OpGlobal)
      continue;

    uint64_t Offset =
        cast<ConstantInt>(
            cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
            ->getZExtValue();

    VTableBits *&BitsPtr = GVToBits[OpGlobal];
    if (!BitsPtr) {
      Bits.emplace_back();
      Bits.back().GV = OpGlobal;
      Bits.back().ObjectSize = M.getDataLayout().getTypeAllocSize(
          OpGlobal->getInitializer()->getType());
      BitsPtr = &Bits.back();
    }
    BitSets[BitSetID].insert({BitsPtr, Offset});
  }
}

bool DevirtModule::tryFindVirtualCallTargets(
    std::vector<VirtualCallTarget> &TargetsForSlot,
    const std::set<BitSetInfo> &BitSetInfos, uint64_t ByteOffset) {
  for (const BitSetInfo &BS : BitSetInfos) {
    if (!BS.Bits->GV->isConstant())
      return false;

    auto Init = dyn_cast<ConstantArray>(BS.Bits->GV->getInitializer());
    if (!Init)
      return false;
    ArrayType *VTableTy = Init->getType();

    uint64_t ElemSize =
        M.getDataLayout().getTypeAllocSize(VTableTy->getElementType());
    uint64_t GlobalSlotOffset = BS.Offset + ByteOffset;
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

    TargetsForSlot.push_back({Fn, &BS});
  }

  // Give up if we couldn't find any targets.
  return !TargetsForSlot.empty();
}

bool DevirtModule::trySingleImplDevirt(
    ArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // See if the program contains a single implementation of this virtual
  // function.
  Function *TheFn = TargetsForSlot[0].Fn;
  for (auto &&Target : TargetsForSlot)
    if (TheFn != Target.Fn)
      return false;

  // If so, update each call site to call that implementation directly.
  for (auto &&VCallSite : CallSites) {
    VCallSite.CS.setCalledFunction(ConstantExpr::getBitCast(
        TheFn, VCallSite.CS.getCalledValue()->getType()));
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
    IntegerType *RetType, ArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // Uniform return value optimization. If all functions return the same
  // constant, replace all calls with that constant.
  uint64_t TheRetVal = TargetsForSlot[0].RetVal;
  for (const VirtualCallTarget &Target : TargetsForSlot)
    if (Target.RetVal != TheRetVal)
      return false;

  auto TheRetValConst = ConstantInt::get(RetType, TheRetVal);
  for (auto Call : CallSites)
    Call.replaceAndErase(TheRetValConst);
  return true;
}

bool DevirtModule::tryUniqueRetValOpt(
    unsigned BitWidth, ArrayRef<VirtualCallTarget> TargetsForSlot,
    MutableArrayRef<VirtualCallSite> CallSites) {
  // IsOne controls whether we look for a 0 or a 1.
  auto tryUniqueRetValOptFor = [&](bool IsOne) {
    const BitSetInfo *UniqueBitSet = 0;
    for (const VirtualCallTarget &Target : TargetsForSlot) {
      if (Target.RetVal == (IsOne ? 1 : 0)) {
        if (UniqueBitSet)
          return false;
        UniqueBitSet = Target.BS;
      }
    }

    // We should have found a unique bit set or bailed out by now. We already
    // checked for a uniform return value in tryUniformRetValOpt.
    assert(UniqueBitSet);

    // Replace each call with the comparison.
    for (auto &&Call : CallSites) {
      IRBuilder<> B(Call.CS.getInstruction());
      Value *OneAddr = B.CreateBitCast(UniqueBitSet->Bits->GV, Int8PtrTy);
      OneAddr = B.CreateConstGEP1_64(OneAddr, UniqueBitSet->Offset);
      Value *Cmp = B.CreateICmp(IsOne ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE,
                                Call.VTable, OneAddr);
      Call.replaceAndErase(Cmp);
    }
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

    // Find an allocation offset in bits in all vtables in the bitset.
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

    // Rewrite each call to a load from OffsetByte/OffsetBit.
    for (auto Call : CSByConstantArg.second) {
      IRBuilder<> B(Call.CS.getInstruction());
      Value *Addr = B.CreateConstGEP1_64(Call.VTable, OffsetByte);
      if (BitWidth == 1) {
        Value *Bits = B.CreateLoad(Addr);
        Value *Bit = ConstantInt::get(Int8Ty, 1ULL << OffsetBit);
        Value *BitsAndBit = B.CreateAnd(Bits, Bit);
        auto IsBitSet = B.CreateICmpNE(BitsAndBit, ConstantInt::get(Int8Ty, 0));
        Call.replaceAndErase(IsBitSet);
      } else {
        Value *ValAddr = B.CreateBitCast(Addr, RetType->getPointerTo());
        Value *Val = B.CreateLoad(RetType, ValAddr);
        Call.replaceAndErase(Val);
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

bool DevirtModule::run() {
  Function *BitSetTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::bitset_test));
  if (!BitSetTestFunc || BitSetTestFunc->use_empty())
    return false;

  Function *AssumeFunc = M.getFunction(Intrinsic::getName(Intrinsic::assume));
  if (!AssumeFunc || AssumeFunc->use_empty())
    return false;

  // Find all virtual calls via a virtual table pointer %p under an assumption
  // of the form llvm.assume(llvm.bitset.test(%p, %md)). This indicates that %p
  // points to a vtable in the bitset %md. Group calls by (bitset, offset) pair
  // (effectively the identity of the virtual function) and store to CallSlots.
  DenseSet<Value *> SeenPtrs;
  for (auto I = BitSetTestFunc->use_begin(), E = BitSetTestFunc->use_end();
       I != E;) {
    auto CI = dyn_cast<CallInst>(I->getUser());
    ++I;
    if (!CI)
      continue;

    // Find llvm.assume intrinsics for this llvm.bitset.test call.
    SmallVector<CallInst *, 1> Assumes;
    for (const Use &CIU : CI->uses()) {
      auto AssumeCI = dyn_cast<CallInst>(CIU.getUser());
      if (AssumeCI && AssumeCI->getCalledValue() == AssumeFunc)
        Assumes.push_back(AssumeCI);
    }

    // If we found any, search for virtual calls based on %p and add them to
    // CallSlots.
    if (!Assumes.empty()) {
      Metadata *BitSet =
          cast<MetadataAsValue>(CI->getArgOperand(1))->getMetadata();
      Value *Ptr = CI->getArgOperand(0)->stripPointerCasts();
      if (SeenPtrs.insert(Ptr).second)
        findLoadCallsAtConstantOffset(BitSet, Ptr, 0, CI->getArgOperand(0));
    }

    // We no longer need the assumes or the bitset test.
    for (auto Assume : Assumes)
      Assume->eraseFromParent();
    // We can't use RecursivelyDeleteTriviallyDeadInstructions here because we
    // may use the vtable argument later.
    if (CI->use_empty())
      CI->eraseFromParent();
  }

  // Rebuild llvm.bitsets metadata into a map for easy lookup.
  std::vector<VTableBits> Bits;
  DenseMap<Metadata *, std::set<BitSetInfo>> BitSets;
  buildBitSets(Bits, BitSets);
  if (BitSets.empty())
    return true;

  // For each (bitset, offset) pair:
  bool DidVirtualConstProp = false;
  for (auto &S : CallSlots) {
    // Search each of the vtables in the bitset for the virtual function
    // implementation at offset S.first.ByteOffset, and add to TargetsForSlot.
    std::vector<VirtualCallTarget> TargetsForSlot;
    if (!tryFindVirtualCallTargets(TargetsForSlot, BitSets[S.first.BitSetID],
                                   S.first.ByteOffset))
      continue;

    if (trySingleImplDevirt(TargetsForSlot, S.second))
      continue;

    DidVirtualConstProp |= tryVirtualConstProp(TargetsForSlot, S.second);
  }

  // Rebuild each global we touched as part of virtual constant propagation to
  // include the before and after bytes.
  if (DidVirtualConstProp)
    for (VTableBits &B : Bits)
      rebuildGlobal(B);

  return true;
}
