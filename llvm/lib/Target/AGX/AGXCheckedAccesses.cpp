//===-- AGXCheckedAccesses.cpp - Checked memory accesses ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The agx-checked feature (llvm/docs/AGXUsage.rst, "Checked mode"): every
// load, store and atomic of a kernel is checked before it runs, and the first
// bad one is written to an error record that the runtime reads after the
// launch.
//
// Device accesses are checked for null (an address below 4096), alignment (a
// multiple of the access size, up to 16 bytes) and, when the pointer is based
// on a kernel argument, a pointer field of a by-reference argument or a module
// global, for lying in that base's allocation. The bounds of argument-based
// pointers ("slots") are hidden kernel arguments the runtime fills at launch.
// Threadgroup accesses are checked against the launch's threadgroup memory
// size, also a hidden argument, and stack accesses with variable offsets
// against their object.
//
// A failing thread claims the error record with a compare-and-swap, writes
// it and returns from the kernel. Each kernel first returns if the record is
// already claimed.
//
// The pass runs after inlining and InferAddressSpaces, and before generic
// pointers are lowered and kernel arguments are promoted (whose argument
// table leaves the hidden arguments out).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "agx-checked-accesses"

namespace {

// Error kinds of the error record (AGXUsage.rst).
enum : unsigned {
  KindOutOfBounds = 1,
  KindNull = 2,
  KindMisaligned = 3,
  KindThreadgroup = 4,
  KindStack = 5,
  KindStore = 1 << 8,
  KindAtomic = 1 << 9,
};

constexpr uint64_t NullLimit = 4096;
constexpr unsigned MaxSlots = 16;
constexpr unsigned NoOffset = ~0u;

class AGXCheckedAccesses : public ModulePass {
public:
  static char ID;
  const AGXTargetMachine *TM;
  explicit AGXCheckedAccesses(const AGXTargetMachine *TM = nullptr)
      : ModulePass(ID), TM(TM) {}

  StringRef getPassName() const override {
    return "AGX Checked Memory Accesses";
  }
  bool runOnModule(Module &M) override;
};

struct Access {
  Instruction *I;
  BasicBlock *Block; // before any check splits it
  Value *Ptr; // only before the hidden arguments are added
  uint64_t Size;
  unsigned Flags; // KindStore / KindAtomic
};

/// Where a device pointer comes from.
struct Base {
  enum { Unknown, Slot, Global } K = Unknown;
  unsigned Arg = 0, Offset = NoOffset; // Slot
  const GlobalVariable *GV = nullptr;   // Global
};

struct Kernel {
  Function *F;
  SmallVector<Access, 32> Accesses;
  SmallVector<std::pair<unsigned, unsigned>, 8> Slots; // (argument, offset)
  DenseMap<std::pair<unsigned, unsigned>, unsigned> SlotIndex;
};

} // namespace

char AGXCheckedAccesses::ID = 0;

INITIALIZE_PASS(AGXCheckedAccesses, DEBUG_TYPE, "AGX Checked Memory Accesses",
                false, false)

ModulePass *llvm::createAGXCheckedAccessesPass(const AGXTargetMachine *TM) {
  return new AGXCheckedAccesses(TM);
}

static Base findBase(const Value *Ptr, const DataLayout &DL, unsigned Depth = 0);

/// The base of a 64-bit integer that becomes a pointer: front ends like ZLUDA
/// load PTX pointer parameters as i64 and add offsets before inttoptr.
static Base findIntBase(const Value *V, const DataLayout &DL, unsigned Depth) {
  if (Depth > 16)
    return Base();
  if (const auto *L = dyn_cast<LoadInst>(V)) {
    Base B;
    int64_t Off = 0;
    const Value *P =
        GetPointerBaseWithConstantOffset(L->getPointerOperand(), Off, DL);
    const auto *A = dyn_cast<Argument>(P);
    if (A && A->getParamByRefType() && Off >= 0 &&
        DL.getTypeStoreSize(L->getType()) == 8 &&
        uint64_t(Off) + 8 <= DL.getTypeAllocSize(A->getParamByRefType())) {
      B.K = Base::Slot;
      B.Arg = A->getArgNo();
      B.Offset = unsigned(Off);
    }
    return B;
  }
  if (const auto *A = dyn_cast<Argument>(V)) {
    Base B;
    if (DL.getTypeStoreSize(A->getType()) == 8 && !A->getType()->isPointerTy()) {
      B.K = Base::Slot;
      B.Arg = A->getArgNo();
    }
    return B;
  }
  if (const auto *PI = dyn_cast<PtrToIntOperator>(V))
    return findBase(PI->getPointerOperand(), DL, Depth + 1);
  if (const auto *BO = dyn_cast<BinaryOperator>(V)) {
    if (BO->getOpcode() == Instruction::Add ||
        (BO->getOpcode() == Instruction::Or &&
         cast<PossiblyDisjointInst>(BO)->isDisjoint())) {
      Base L = findIntBase(BO->getOperand(0), DL, Depth + 1);
      Base R = findIntBase(BO->getOperand(1), DL, Depth + 1);
      // base + offset: exactly one side is a pointer base
      if (L.K != Base::Unknown && R.K == Base::Unknown)
        return L;
      if (R.K != Base::Unknown && L.K == Base::Unknown)
        return R;
    }
    if (BO->getOpcode() == Instruction::Sub) {
      Base L = findIntBase(BO->getOperand(0), DL, Depth + 1);
      if (L.K != Base::Unknown &&
          findIntBase(BO->getOperand(1), DL, Depth + 1).K == Base::Unknown)
        return L;
    }
  }
  return Base();
}

static Base findBase(const Value *Ptr, const DataLayout &DL, unsigned Depth) {
  Base B;
  if (Depth > 16)
    return B;
  const Value *U = getUnderlyingObject(Ptr, /*MaxLookup=*/0);
  // A field of an aggregate built with insertvalue: MLIR's memref
  // descriptors hold the pointer arguments of a kernel.
  if (const auto *EV = dyn_cast<ExtractValueInst>(U))
    if (Value *F = FindInsertedValue(
            const_cast<Value *>(EV->getAggregateOperand()), EV->getIndices()))
      if (F != EV)
        return findBase(F, DL, Depth + 1);
  if (const auto *I2P = dyn_cast<IntToPtrInst>(U))
    return findIntBase(I2P->getOperand(0), DL, Depth + 1);
  if (const auto *CE = dyn_cast<ConstantExpr>(U))
    if (CE->getOpcode() == Instruction::IntToPtr)
      return findIntBase(CE->getOperand(0), DL, Depth + 1);
  if (const auto *A = dyn_cast<Argument>(U)) {
    if (A->getType()->isPointerTy() && !A->getParamByRefType()) {
      B.K = Base::Slot;
      B.Arg = A->getArgNo();
    }
    return B;
  }
  if (const auto *L = dyn_cast<LoadInst>(U)) {
    // A pointer field of a by-reference argument (PTX .param pointers).
    int64_t Off = 0;
    const Value *P =
        GetPointerBaseWithConstantOffset(L->getPointerOperand(), Off, DL);
    const auto *A = dyn_cast<Argument>(P);
    if (A && A->getParamByRefType() && Off >= 0 &&
        DL.getTypeStoreSize(L->getType()) == 8 &&
        uint64_t(Off) + 8 <= DL.getTypeAllocSize(A->getParamByRefType())) {
      B.K = Base::Slot;
      B.Arg = A->getArgNo();
      B.Offset = unsigned(Off);
    }
    return B;
  }
  if (const auto *GV = dyn_cast<GlobalVariable>(U))
    if (!GV->isDeclaration() && GV->getValueType()->isSized()) {
      B.K = Base::Global;
      B.GV = GV;
    }
  return B;
}

/// The byte offset of Ptr in the alloca it is based on, if it is one and the
/// offset is not constant. Emitted before IP.
static Value *stackOffset(Value *Ptr, const DataLayout &DL, IRBuilder<> &B,
                          AllocaInst *&Alloca) {
  SmallVector<GEPOperator *, 4> GEPs;
  Value *V = Ptr;
  while (true) {
    V = V->stripPointerCasts();
    if (auto *G = dyn_cast<GEPOperator>(V)) {
      GEPs.push_back(G);
      V = G->getPointerOperand();
      continue;
    }
    break;
  }
  Alloca = dyn_cast<AllocaInst>(V);
  if (!Alloca || GEPs.empty() ||
      llvm::all_of(GEPs, [](GEPOperator *G) { return G->hasAllConstantIndices(); }))
    return nullptr;
  Type *IdxTy = DL.getIndexType(Ptr->getType());
  Value *Off = ConstantInt::get(IdxTy, 0);
  for (GEPOperator *G : GEPs) {
    // Walk the indices like emitGEPOffset, in the index type.
    gep_type_iterator GTI = gep_type_begin(G);
    for (Use &Idx : G->indices()) {
      Value *I = B.CreateSExtOrTrunc(Idx, IdxTy);
      if (StructType *ST = GTI.getStructTypeOrNull()) {
        uint64_t FO = DL.getStructLayout(ST)
                          ->getElementOffset(cast<ConstantInt>(Idx)->getZExtValue());
        Off = B.CreateAdd(Off, ConstantInt::get(IdxTy, FO));
      } else {
        uint64_t Stride = GTI.getSequentialElementStride(DL);
        Off = B.CreateAdd(Off, B.CreateMul(I, ConstantInt::get(IdxTy, Stride)));
      }
      ++GTI;
    }
  }
  return Off;
}

/// The frames of a debug location, innermost first.
static std::string siteFrames(const Instruction *I, const Function &K) {
  std::string S;
  raw_string_ostream OS(S);
  const DILocation *Loc = I->getDebugLoc();
  if (!Loc) {
    OS << K.getName();
    return S;
  }
  for (bool First = true; Loc; Loc = Loc->getInlinedAt(), First = false) {
    if (!First)
      OS << '\t';
    StringRef Name = Loc->getScope()->getSubprogram()
                         ? Loc->getScope()->getSubprogram()->getName()
                         : StringRef();
    OS << (Name.empty() ? K.getName() : Name) << ' ' << Loc->getFilename()
       << ':' << Loc->getLine() << ':' << Loc->getColumn();
  }
  return S;
}

static const char *accessName(unsigned Flags) {
  return Flags & KindAtomic ? "atomic" : Flags & KindStore ? "store" : "load";
}

/// The kernel with its hidden arguments: record, threadgroup bytes, then the
/// two bounds of each slot.
static Function *addHiddenArgs(Kernel &K) {
  Function &F = *K.F;
  LLVMContext &Ctx = F.getContext();
  SmallVector<Type *, 16> Params(F.getFunctionType()->params());
  AttributeList Attrs = F.getAttributes();
  SmallVector<AttributeSet, 16> ParamAttrs;
  for (unsigned I = 0; I < F.arg_size(); ++I)
    ParamAttrs.push_back(Attrs.getParamAttrs(I));
  auto Hidden = [&](Type *T, StringRef Role) {
    Params.push_back(T);
    ParamAttrs.push_back(
        AttributeSet::get(Ctx, {Attribute::get(Ctx, "agx-checked", Role)}));
  };
  Hidden(PointerType::get(Ctx, AGX::DEVICE_ADDRESS), "record");
  Hidden(Type::getInt32Ty(Ctx), "threadgroup-bytes");
  for (unsigned S = 0; S < K.Slots.size(); ++S) {
    Hidden(Type::getInt64Ty(Ctx), ("lo" + Twine(S)).str());
    Hidden(Type::getInt64Ty(Ctx), ("hi" + Twine(S)).str());
  }
  FunctionType *NFTy = FunctionType::get(F.getReturnType(), Params, false);
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());
  NF->copyAttributesFrom(&F);
  NF->setAttributes(AttributeList::get(Ctx, Attrs.getFnAttrs(),
                                       Attrs.getRetAttrs(), ParamAttrs));
  NF->copyMetadata(&F, 0);
  NF->takeName(&F);
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->splice(NF->begin(), &F);
  for (Argument &A : F.args()) {
    Argument *NA = NF->getArg(A.getArgNo());
    NA->takeName(&A);
    A.replaceAllUsesWith(NA);
  }
  std::string Desc;
  raw_string_ostream OS(Desc);
  for (auto [I, S] : enumerate(K.Slots))
    OS << (I ? "," : "") << S.first << ':' << int64_t(int32_t(S.second));
  NF->addFnAttr("agx-checked-slots", Desc);
  F.eraseFromParent();
  K.F = NF;
  return NF;
}

static Value *checkedPointer(Instruction *I) {
  if (Value *P = getLoadStorePointerOperand(I))
    return P;
  if (auto *R = dyn_cast<AtomicRMWInst>(I))
    return R->getPointerOperand();
  return cast<AtomicCmpXchgInst>(I)->getPointerOperand();
}

static bool instrument(Kernel &K, std::string &Sites, unsigned &NextSite) {
  Function &F = *K.F;
  Module &M = *F.getParent();
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = F.getContext();
  Type *I32 = Type::getInt32Ty(Ctx), *I64 = Type::getInt64Ty(Ctx);

  // Bases first: the slots decide the hidden arguments.
  SmallVector<Base, 32> Bases;
  for (Access &A : K.Accesses) {
    unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
    Base B;
    if (AS == AGX::DEVICE_ADDRESS || AS == AGX::GENERIC_ADDRESS)
      B = findBase(A.Ptr, DL);
    if (B.K == Base::Slot) {
      auto Key = std::make_pair(B.Arg, B.Offset);
      if (!K.SlotIndex.count(Key)) {
        if (K.Slots.size() == MaxSlots)
          B.K = Base::Unknown;
        else {
          K.SlotIndex[Key] = K.Slots.size();
          K.Slots.push_back(Key);
        }
      }
    }
    Bases.push_back(B);
  }
  unsigned NumSource = F.arg_size();
  Function &NF = *addHiddenArgs(K);
  Argument *Record = NF.getArg(NumSource);
  Argument *TGBytes = NF.getArg(NumSource + 1);
  auto Lo = [&](unsigned S) { return NF.getArg(NumSource + 2 + 2 * S); };
  auto Hi = [&](unsigned S) { return NF.getArg(NumSource + 3 + 2 * S); };

  // The report block: claim the record, fill it, return.
  BasicBlock *Report = BasicBlock::Create(Ctx, "checked.report", &NF);
  BasicBlock *Fill = BasicBlock::Create(Ctx, "checked.fill", &NF);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "checked.exit", &NF);
  // The kind is worked out here, off the hot path: a failed device check is
  // a null access, a misaligned one, or else out of bounds.
  IRBuilder<> RB(Report);
  PHINode *PClass = RB.CreatePHI(I32, 0, "class");
  PHINode *PAddr = RB.CreatePHI(I64, 0, "addr");
  PHINode *PMask = RB.CreatePHI(I32, 0, "alignmask");
  PHINode *PAlign = RB.CreatePHI(I32, 0, "alignbits"); // low bits checked
  PHINode *PSize = RB.CreatePHI(I32, 0, "size");
  PHINode *PSite = RB.CreatePHI(I32, 0, "site");
  Value *Claim = RB.CreateAtomicCmpXchg(
      Record, RB.getInt32(0), RB.getInt32(1), MaybeAlign(4),
      AtomicOrdering::Monotonic, AtomicOrdering::Monotonic);
  RB.CreateCondBr(RB.CreateExtractValue(Claim, 1), Fill, Exit);
  IRBuilder<> FB(Fill);
  auto Field = [&](uint64_t Off, Value *V) {
    FB.CreateAlignedStore(V, FB.CreateConstInBoundsGEP1_64(FB.getInt8Ty(),
                                                           Record, Off),
                          Align(4));
  };
  Value *IsDevice = FB.CreateICmpEQ(FB.CreateAnd(PClass, FB.getInt32(0xff)),
                                    FB.getInt32(KindOutOfBounds));
  Value *DevKind = FB.CreateSelect(
      FB.CreateICmpULT(PAddr, FB.getInt64(NullLimit)), FB.getInt32(KindNull),
      FB.CreateSelect(
          FB.CreateICmpNE(FB.CreateAnd(PAlign, PMask), FB.getInt32(0)),
          FB.getInt32(KindMisaligned), FB.getInt32(KindOutOfBounds)));
  Value *PKind = FB.CreateOr(
      FB.CreateSelect(IsDevice, DevKind,
                      FB.CreateAnd(PClass, FB.getInt32(0xff))),
      FB.CreateAnd(PClass, FB.getInt32(~0xffu)));
  Field(4, PKind);
  Field(8, PAddr);
  Field(16, PSize);
  Field(20, PSite);
  Intrinsic::ID TID[3] = {Intrinsic::agx_thread_position_in_threadgroup_x,
                          Intrinsic::agx_thread_position_in_threadgroup_y,
                          Intrinsic::agx_thread_position_in_threadgroup_z};
  Intrinsic::ID GID[3] = {Intrinsic::agx_threadgroup_position_in_grid_x,
                          Intrinsic::agx_threadgroup_position_in_grid_y,
                          Intrinsic::agx_threadgroup_position_in_grid_z};
  for (unsigned D = 0; D < 3; ++D) {
    Field(24 + 4 * D, FB.CreateIntrinsic(TID[D], {}));
    Field(36 + 4 * D, FB.CreateIntrinsic(GID[D], {}));
  }
  FB.CreateBr(Exit);
  ReturnInst::Create(Ctx, Exit);

  // What each check covers. Accesses of one block that differ by constant
  // offsets from the same base and do the same kind of access share one check
  // over their whole byte range, made before the first of them; their
  // alignment is checked once when every offset keeps it. So repeated and
  // unrolled accesses cost one check.
  struct Check {
    Instruction *At;
    Value *From;     // the pointer the byte range is relative to
    int64_t Lo, Hi;  // [Lo, Hi) from From
    uint32_t Mask;   // alignment checked on From: each member's offset is a
                     // multiple of its own alignment
    unsigned Flags, AS;
    Base B;
  };
  SmallVector<Check, 32> Checks;
  DenseMap<std::tuple<BasicBlock *, Value *, unsigned>, unsigned> Groups;
  // The checks of each (block, base, address space), to skip an access
  // whose bytes an earlier check already covers, whatever its kind.
  DenseMap<std::tuple<BasicBlock *, Value *, unsigned>, SmallVector<unsigned, 2>>
      Covering;
  for (auto [A, B] : zip(K.Accesses, Bases)) {
    // The pointer operand again: the kernel's arguments have been replaced.
    A.Ptr = checkedPointer(A.I);
    unsigned AS = A.Ptr->getType()->getPointerAddressSpace();
    bool Device = AS == AGX::DEVICE_ADDRESS || AS == AGX::GENERIC_ADDRESS;
    if (!Device && AS != AGX::THREADGROUP_ADDRESS &&
        AS != AGX::PRIVATE_ADDRESS)
      continue;
    int64_t Off = 0;
    Value *From = A.Ptr;
    if (AS != AGX::PRIVATE_ADDRESS)
      From = GetPointerBaseWithConstantOffset(A.Ptr, Off, DL);
    uint64_t AlignTo = std::min<uint64_t>(A.Size, 16);
    uint32_t Mask =
        Device && isPowerOf2_64(AlignTo) && AlignTo > 1 ? AlignTo - 1 : 0;
    auto Where = std::make_tuple(A.Block, From, AS);
    if (llvm::any_of(Covering[Where], [&](unsigned I) {
          const Check &C = Checks[I];
          // (a check's range only grows as accesses merge into it)
          return C.Lo <= Off && Off + int64_t(A.Size) <= C.Hi &&
                 (Off & Mask) == 0 && C.Mask >= Mask;
        }))
      continue;
    auto Key = std::make_tuple(A.Block, From, (AS << 16) | A.Flags);
    auto G = Groups.find(Key);
    if (G != Groups.end() && (Off & Mask) == 0) {
      Check &C = Checks[G->second];
      C.Lo = std::min(C.Lo, Off);
      C.Hi = std::max<int64_t>(C.Hi, Off + A.Size);
      C.Mask = std::max(C.Mask, Mask);
      continue;
    }
    Checks.push_back({A.I, From, Off, Off + int64_t(A.Size), Mask, A.Flags,
                      AS, B});
    Covering[Where].push_back(Checks.size() - 1);
    if ((Off & Mask) == 0)
      Groups[Key] = Checks.size() - 1;
  }

  for (Check &C : Checks) {
    IRBuilder<> IB(C.At);
    uint64_t Size = C.Hi - C.Lo;
    Value *Start = C.Lo ? IB.CreateConstGEP1_64(IB.getInt8Ty(), C.From, C.Lo)
                        : C.From;
    Value *Bad = nullptr, *Kind = nullptr, *Addr = nullptr;
    Value *AlignBits = IB.getInt32(0);
    if (C.AS == AGX::DEVICE_ADDRESS || C.AS == AGX::GENERIC_ADDRESS) {
      Addr = IB.CreatePtrToInt(Start, I64);
      Kind = IB.getInt32(KindOutOfBounds);
      if (C.Mask) {
        AlignBits = IB.CreateTrunc(
            C.Lo ? IB.CreatePtrToInt(C.From, I64) : Addr, I32);
        Bad = IB.CreateICmpNE(IB.CreateAnd(AlignBits, IB.getInt32(C.Mask)),
                              IB.getInt32(0));
      }
      // A pointer outside every allocation has empty bounds, so the bounds
      // check also catches null; without bounds, null is checked directly.
      Value *Range;
      if (C.B.K == Base::Slot) {
        unsigned S = K.SlotIndex.lookup({C.B.Arg, C.B.Offset});
        Value *End = IB.CreateAdd(Addr, IB.getInt64(Size));
        Range = IB.CreateOr(IB.CreateICmpULT(Addr, Lo(S)),
                            IB.CreateICmpUGT(End, Hi(S)));
      } else if (C.B.K == Base::Global) {
        // Constant bounds: one compare of the offset in the global.
        uint64_t Bytes = DL.getTypeAllocSize(C.B.GV->getValueType());
        Value *GVAddr =
            IB.CreatePtrToInt(const_cast<GlobalVariable *>(C.B.GV), I64);
        Range = Size > Bytes ? IB.getTrue()
                             : IB.CreateICmpUGT(IB.CreateSub(Addr, GVAddr),
                                                IB.getInt64(Bytes - Size));
      } else {
        Range = IB.CreateICmpULT(Addr, IB.getInt64(NullLimit));
      }
      Bad = Bad ? IB.CreateOr(Bad, Range) : Range;
    } else if (C.AS == AGX::THREADGROUP_ADDRESS) {
      // Threadgroup offsets are 32-bit. A static array bounds its accesses;
      // others are bounded by the launch's threadgroup memory.
      Type *OffTy = IB.getIntNTy(DL.getPointerSizeInBits(C.AS));
      Value *Off = IB.CreatePtrToInt(Start, OffTy);
      Addr = IB.CreateZExt(Off, I64);
      Kind = IB.getInt32(KindThreadgroup);
      const auto *GV =
          dyn_cast<GlobalVariable>(getUnderlyingObject(C.From, 0));
      uint64_t Bytes =
          GV && GV->getValueType()->isSized()
              ? DL.getTypeAllocSize(GV->getValueType()).getKnownMinValue()
              : 0;
      if (Bytes) {
        Value *GVOff =
            IB.CreatePtrToInt(const_cast<GlobalVariable *>(GV), OffTy);
        Bad = Size > Bytes ? IB.getTrue()
                           : IB.CreateICmpUGT(IB.CreateSub(Off, GVOff),
                                              ConstantInt::get(OffTy, Bytes - Size));
      } else {
        Value *Limit = IB.CreateZExtOrTrunc(TGBytes, OffTy);
        Bad = IB.CreateOr(
            IB.CreateICmpUGT(Off, Limit),
            IB.CreateICmpUGT(IB.CreateAdd(Off, ConstantInt::get(OffTy, Size)),
                             Limit));
      }
    } else {
      AllocaInst *Alloca = nullptr;
      Value *Off = stackOffset(Start, DL, IB, Alloca);
      std::optional<TypeSize> Bytes =
          Alloca ? Alloca->getAllocationSize(DL) : std::nullopt;
      if (!Off || !Bytes || Bytes->isScalable())
        continue;
      Value *Off64 = IB.CreateSExt(Off, I64);
      Addr = Off64;
      Bad = IB.CreateOr(
          IB.CreateICmpSLT(Off64, IB.getInt64(0)),
          IB.CreateICmpSGT(IB.CreateAdd(Off64, IB.getInt64(Size)),
                           IB.getInt64(Bytes->getFixedValue())));
      Kind = IB.getInt32(KindStack);
    }
    unsigned Site = NextSite++;
    raw_string_ostream(Sites) << Site << '\t' << accessName(C.Flags) << ' '
                              << Size << '\t' << siteFrames(C.At, NF) << '\n';
    Kind = IB.CreateOr(Kind, IB.getInt32(C.Flags));
    BasicBlock *Here = C.At->getParent();
    BasicBlock *Cont = SplitBlock(Here, C.At);
    Here->getTerminator()->eraseFromParent();
    IRBuilder<> BB(Here);
    BB.CreateCondBr(Bad, Report, Cont);
    PClass->addIncoming(Kind, Here);
    PAddr->addIncoming(Addr, Here);
    PMask->addIncoming(BB.getInt32(C.Mask), Here);
    PAlign->addIncoming(AlignBits, Here);
    PSize->addIncoming(BB.getInt32(Size), Here);
    PSite->addIncoming(BB.getInt32(Site), Here);
  }

  // Return at once when an earlier launch recorded an error. The allocas
  // stay in the entry block.
  BasicBlock &Entry = NF.getEntryBlock();
  BasicBlock::iterator IP = Entry.getFirstInsertionPt();
  while (isa<AllocaInst>(*IP))
    ++IP;
  BasicBlock *Body = SplitBlock(&Entry, IP);
  Entry.getTerminator()->eraseFromParent();
  IRBuilder<> EB(&Entry);
  // A plain load: the record was written by earlier, finished commands.
  LoadInst *Claimed = EB.CreateAlignedLoad(I32, Record, Align(4), "claimed");
  EB.CreateCondBr(EB.CreateICmpNE(Claimed, EB.getInt32(0)), Exit, Body);

  if (PClass->getNumIncomingValues() == 0)
    // Nothing to check: the report blocks are unreachable.
    DeleteDeadBlocks({Report, Fill});
  return true;
}

bool AGXCheckedAccesses::runOnModule(Module &M) {
  SmallVector<Kernel, 8> Kernels;
  for (Function &F : M) {
    if (F.getCallingConv() != CallingConv::AGX_KERNEL || F.isDeclaration() ||
        !F.use_empty() || F.isVarArg() ||
        !TM->getSubtarget<AGXSubtarget>(F).hasChecked())
      continue;
    Kernel K{&F, {}, {}, {}};
    for (Instruction &I : instructions(F)) {
      const DataLayout &DL = M.getDataLayout();
      if (auto *L = dyn_cast<LoadInst>(&I))
        K.Accesses.push_back({L, L->getParent(), L->getPointerOperand(),
                              DL.getTypeStoreSize(L->getType()), 0});
      else if (auto *S = dyn_cast<StoreInst>(&I))
        K.Accesses.push_back(
            {S, S->getParent(), S->getPointerOperand(),
             DL.getTypeStoreSize(S->getValueOperand()->getType()), KindStore});
      else if (auto *R = dyn_cast<AtomicRMWInst>(&I))
        K.Accesses.push_back({R, R->getParent(), R->getPointerOperand(),
                              DL.getTypeStoreSize(R->getType()), KindAtomic});
      else if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
        K.Accesses.push_back(
            {X, X->getParent(), X->getPointerOperand(),
             DL.getTypeStoreSize(X->getCompareOperand()->getType()),
             KindAtomic});
    }
    Kernels.push_back(std::move(K));
  }
  if (Kernels.empty())
    return false;
  std::string Sites;
  unsigned NextSite = 1;
  for (Kernel &K : Kernels)
    instrument(K, Sites, NextSite);

  // The check site table, in the global data the runtime can read.
  Constant *Text = ConstantDataArray::getString(M.getContext(), Sites);
  auto *GV = new GlobalVariable(M, Text->getType(), /*isConstant=*/true,
                                GlobalValue::ExternalLinkage, Text,
                                "__agx_checked_sites", nullptr,
                                GlobalValue::NotThreadLocal,
                                AGX::DEVICE_ADDRESS);
  appendToCompilerUsed(M, {GV});
  return true;
}
