//===-- GPUSan.cpp - GPU sanitizer ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/GPUSan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "gpusan"

cl::opt<bool> UseTags(
    "gpusan-use-tags",
    cl::desc(
        "Use tags to detect use after if the number of allocations is large"),
    cl::init(false));

namespace llvm {

struct LocationInfoTy {
  uint64_t LineNo = 0;
  uint64_t ColumnNo = 0;
  uint64_t ParentIdx = -1;
  StringRef FileName;
  StringRef FunctionName;
  bool operator==(const LocationInfoTy &RHS) const {
    return LineNo == RHS.LineNo && ColumnNo == RHS.ColumnNo &&
           FileName == RHS.FileName && FunctionName == RHS.FunctionName;
  }
};
template <> struct DenseMapInfo<LocationInfoTy *> {
  static LocationInfoTy EmptyKey;
  static LocationInfoTy TombstoneKey;
  static inline LocationInfoTy *getEmptyKey() { return &EmptyKey; }

  static inline LocationInfoTy *getTombstoneKey() { return &TombstoneKey; }

  static unsigned getHashValue(const LocationInfoTy *LI) {
    unsigned Hash = DenseMapInfo<uint64_t>::getHashValue(LI->LineNo);
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<uint64_t>::getHashValue(LI->ColumnNo));
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<StringRef>::getHashValue(LI->FileName));
    Hash = detail::combineHashValue(
        Hash, DenseMapInfo<StringRef>::getHashValue(LI->FunctionName));
    return Hash;
  }

  static bool isEqual(const LocationInfoTy *LHS, const LocationInfoTy *RHS) {
    return *LHS == *RHS;
  }
};
LocationInfoTy DenseMapInfo<LocationInfoTy *>::EmptyKey =
    LocationInfoTy{(uint64_t)-1};
LocationInfoTy DenseMapInfo<LocationInfoTy *>::TombstoneKey =
    LocationInfoTy{(uint64_t)-2};
} // namespace llvm

namespace {

enum PtrOrigin {
  UNKNOWN,
  LOCAL,
  GLOBAL,
  SYSTEM,
  NONE,
};

static std::string getSuffix(PtrOrigin PO) {
  switch (PO) {
  case UNKNOWN:
    return "";
  case LOCAL:
    return "_local";
  case GLOBAL:
    return "_global";
  default:
    break;
  }
  llvm_unreachable("Bad pointer origin!");
}

static StringRef prettifyFunctionName(StringSaver &SS, StringRef Name) {
  if (Name.ends_with(".internalized"))
    return SS.save(Name.drop_back(sizeof("internalized")) + " (internalized)");
  if (!Name.starts_with("__omp_offloading_"))
    return Name;
  Name = Name.drop_front(sizeof("__omp_offloading_"));
  auto It = Name.find_first_of("_");
  if (It != StringRef::npos && It + 1 < Name.size())
    Name = Name.drop_front(It + 1);
  It = Name.find_first_of("_");
  if (It != StringRef::npos && It + 1 < Name.size())
    Name = Name.drop_front(It + 1);
  if (Name.ends_with("_debug__"))
    Name = Name.drop_back(sizeof("debug__"));
  if (Name.ends_with("_debug___omp_outlined_debug__"))
    Name = Name.drop_back(sizeof("debug___omp_outlined_debug__"));
  It = Name.find_last_of("_");
  if (It == StringRef::npos || It + 1 >= Name.size())
    return Name;
  if (Name[It + 1] != 'l')
    return Name;
  int64_t KernelLineNo = 0;
  Name.take_back(Name.size() - It -
                 /* '_' and 'l' */ 2)
      .getAsInteger(10, KernelLineNo);
  if (KernelLineNo)
    Name = SS.save("omp target (" + Name.take_front(It).str() + ":" +
                   std::to_string(KernelLineNo) + ")");
  return Name;
}

class GPUSanImpl final {
public:
  GPUSanImpl(Module &M, FunctionAnalysisManager &FAM)
      : M(M), FAM(FAM), Ctx(M.getContext()) {}

  bool instrument();

private:
  bool instrumentGlobals();
  bool instrumentFunction(Function &Fn);
  Value *instrumentAllocation(Instruction &I, Value &Size, FunctionCallee Fn,
                              PtrOrigin PO);
  Value *instrumentAllocaInst(LoopInfo &LI, AllocaInst &AI);
  void instrumentAccess(LoopInfo &LI, Instruction &I, int PtrIdx,
                        Type &AccessTy, bool IsRead);
  void instrumentLoadInst(LoopInfo &LI, LoadInst &LoadI);
  void instrumentStoreInst(LoopInfo &LI, StoreInst &StoreI);
  void instrumentGEPInst(LoopInfo &LI, GetElementPtrInst &GEP);
  bool instrumentCallInst(LoopInfo &LI, CallInst &CI);
  void
  instrumentReturns(SmallVectorImpl<std::pair<AllocaInst *, Value *>> &Allocas,
                    SmallVectorImpl<ReturnInst *> &Returns);

  Value *getPC(IRBuilder<> &IRB);
  Value *getFunctionName(IRBuilder<> &IRB);
  Value *getFileName(IRBuilder<> &IRB);
  Value *getLineNo(IRBuilder<> &IRB);

  void getAllocationInfo(Function &Fn, PtrOrigin PO, Value &Object,
                         Value *&Start, Value *&Length, Value *&Tag);
  PtrOrigin getPtrOrigin(LoopInfo &LI, Value *Ptr,
                         const Value **Object = nullptr);

  FunctionCallee getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
      Function *F = cast<Function>(FC.getCallee());
    }
    return FC;
  }

  PointerType *getPtrTy(PtrOrigin PO) {
    if (PO == PtrOrigin::LOCAL)
      return PointerType::get(Ctx, 5);
    return PtrTy;
  }

  FunctionCallee getNewFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(NewFn[PO], "ompx_new" + getSuffix(PO), getPtrTy(PO),
                         {getPtrTy(PO), Int64Ty, Int64Ty, Int64Ty});
  }
  FunctionCallee getFreeFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(FreeFn[PO], "ompx_free" + getSuffix(PO), VoidTy,
                         {getPtrTy(PO), Int64Ty});
  }
  FunctionCallee getFreeNLocalFn() {
    return getOrCreateFn(FreeNLocalFn, "ompx_free_local_n", VoidTy, {Int32Ty});
  }
  FunctionCallee getCheckFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(CheckFn[PO], "ompx_check" + getSuffix(PO),
                         getPtrTy(PO),
                         {getPtrTy(PO), Int64Ty, Int64Ty, Int64Ty});
  }
  FunctionCallee getCheckWithBaseFn(PtrOrigin PO) {
    assert(PO >= LOCAL && PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(CheckWithBaseFn[PO],
                         "ompx_check_with_base" + getSuffix(PO), getPtrTy(PO),
                         {getPtrTy(PO), getPtrTy(PO), Int64Ty, Int32Ty, Int64Ty,
                          Int64Ty, Int64Ty});
  }
  FunctionCallee getAllocationInfoFn(PtrOrigin PO) {
    assert(PO >= LOCAL && PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(
        AllocationInfoFn[PO], "ompx_get_allocation_info" + getSuffix(PO),
        StructType::create({getPtrTy(PO), Int64Ty, Int32Ty}), {getPtrTy(PO)});
  }
  FunctionCallee getGEPFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(GEPFn[PO], "ompx_gep" + getSuffix(PO), getPtrTy(PO),
                         {getPtrTy(PO), Int64Ty, Int64Ty});
  }
  FunctionCallee getUnpackFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(UnpackFn[PO], "ompx_unpack" + getSuffix(PO),
                         getPtrTy(PO), {getPtrTy(PO), Int64Ty});
  }
  FunctionCallee getLifetimeStart() {
    return getOrCreateFn(LifetimeStartFn, "ompx_lifetime_start", VoidTy,
                         {getPtrTy(LOCAL), Int64Ty});
  }
  FunctionCallee getLifetimeEnd() {
    return getOrCreateFn(LifetimeEndFn, "ompx_lifetime_end", VoidTy,
                         {getPtrTy(LOCAL), Int64Ty});
  }
  FunctionCallee getLeakCheckFn() {
    FunctionCallee LeakCheckFn;
    return getOrCreateFn(LeakCheckFn, "ompx_leak_check", VoidTy, {});
  }
  FunctionCallee getThreadIdFn() {
    return getOrCreateFn(ThreadIDFn, "ompx_global_thread_id", Int32Ty, {});
  }

  CallInst *createCall(IRBuilder<> &IRB, FunctionCallee Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "") {
    Calls.push_back(IRB.CreateCall(Callee, Args, Name));
    return Calls.back();
  }
  SmallVector<CallInst *> Calls;

  Module &M;
  FunctionAnalysisManager &FAM;
  LLVMContext &Ctx;
  bool HasAllocas;
  GlobalVariable *LocationsArray;
  SmallSetVector<CallBase *, 16> AmbiguousCalls;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  IntegerType *Int8Ty = Type::getInt8Ty(Ctx);
  IntegerType *Int32Ty = Type::getInt32Ty(Ctx);
  IntegerType *Int64Ty = Type::getInt64Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  FunctionCallee NewFn[3];
  FunctionCallee GEPFn[3];
  FunctionCallee FreeFn[3];
  FunctionCallee CheckFn[3];
  FunctionCallee CheckWithBaseFn[3];
  FunctionCallee AllocationInfoFn[3];
  FunctionCallee UnpackFn[3];
  FunctionCallee LifetimeEndFn;
  FunctionCallee LifetimeStartFn;
  FunctionCallee FreeNLocalFn;
  FunctionCallee ThreadIDFn;

  StringMap<Value *> GlobalStringMap;
  struct AllocationInfoTy {
    Value *Start;
    Value *Length;
    Value *Tag;
  };
  DenseMap<std::pair<Function *, Value *>, AllocationInfoTy> AllocationInfoMap;

  DenseMap<LocationInfoTy *, uint64_t, DenseMapInfo<LocationInfoTy *>>
      LocationMap;

  const std::pair<LocationInfoTy *, uint64_t>
  addLocationInfo(LocationInfoTy *LI, bool &IsNew) {
    auto It = LocationMap.insert({LI, LocationMap.size()});
    IsNew = It.second;
    if (!IsNew)
      delete LI;
    return {It.first->first, It.first->second};
  }

  void buildCallTreeInfo(Function &Fn, LocationInfoTy &LI);
  ConstantInt *getSourceIndex(Instruction &I, LocationInfoTy *LastLI = nullptr);

  uint64_t addString(StringRef S) {
    const auto &It = UniqueStrings.insert({S, ConcatenatedString.size()});
    if (It.second) {
      ConcatenatedString += S;
      ConcatenatedString.push_back('\0');
    }
    return It.first->second;
  };

  void encodeLocationInfo(LocationInfoTy &LI, uint64_t Idx) {
    StringRef FunctionName = LI.FunctionName;
    if (LI.ParentIdx == (decltype(LI.ParentIdx))-1)
      FunctionName = prettifyFunctionName(SS, FunctionName);

    auto FuncIdx = addString(FunctionName);
    auto FileIdx = addString(LI.FileName);
    if (LocationEncoding.size() < (Idx + 1) * 5)
      LocationEncoding.resize((Idx + 1) * 5);
    LocationEncoding[Idx * 5 + 0] = ConstantInt::get(Int64Ty, FuncIdx);
    LocationEncoding[Idx * 5 + 1] = ConstantInt::get(Int64Ty, FileIdx);
    LocationEncoding[Idx * 5 + 2] = ConstantInt::get(Int64Ty, LI.LineNo);
    LocationEncoding[Idx * 5 + 3] = ConstantInt::get(Int64Ty, LI.ColumnNo);
    LocationEncoding[Idx * 5 + 4] = ConstantInt::get(Int64Ty, LI.ParentIdx);
  }

  SmallVector<Constant *> LocationEncoding;
  std::string ConcatenatedString;
  DenseMap<uint64_t, uint64_t> StringIndexMap;
  DenseMap<StringRef, uint64_t> UniqueStrings;

  BumpPtrAllocator BPA;
  StringSaver SS = StringSaver(BPA);
};

} // end anonymous namespace

ConstantInt *GPUSanImpl::getSourceIndex(Instruction &I,
                                        LocationInfoTy *LastLI) {
  LocationInfoTy *LI = new LocationInfoTy();
  auto *DILoc = I.getDebugLoc().get();

  auto FillLI = [&](LocationInfoTy &LI, DILocation &DIL) {
    LI.FileName = DIL.getFilename();
    if (LI.FileName.empty())
      LI.FileName = I.getFunction()->getSubprogram()->getFilename();
    LI.FunctionName = DIL.getSubprogramLinkageName();
    if (LI.FunctionName.empty())
      LI.FunctionName = I.getFunction()->getName();
    LI.LineNo = DIL.getLine();
    LI.ColumnNo = DIL.getColumn();
  };

  DILocation *ParentDILoc = nullptr;
  if (DILoc) {
    FillLI(*LI, *DILoc);
    ParentDILoc = DILoc->getInlinedAt();
  } else {
    LI->FunctionName = I.getFunction()->getName();
  }

  bool IsNew;
  uint64_t Idx;
  std::tie(LI, Idx) = addLocationInfo(LI, IsNew);
  if (LastLI)
    LastLI->ParentIdx = Idx;
  if (!IsNew)
    return ConstantInt::get(Int64Ty, Idx);

  uint64_t CurIdx = Idx;
  LocationInfoTy *CurLI = LI;
  while (ParentDILoc) {
    auto *ParentLI = new LocationInfoTy();
    FillLI(*ParentLI, *ParentDILoc);
    uint64_t ParentIdx;
    std::tie(ParentLI, ParentIdx) = addLocationInfo(ParentLI, IsNew);
    CurLI->ParentIdx = ParentIdx;
    if (!IsNew)
      break;
    encodeLocationInfo(*CurLI, CurIdx);
    CurLI = ParentLI;
    CurIdx = ParentIdx;
    ParentDILoc = ParentDILoc->getInlinedAt();
  }

  Function &Fn = *I.getFunction();
  buildCallTreeInfo(Fn, *CurLI);

  encodeLocationInfo(*CurLI, CurIdx);

  return ConstantInt::get(Int64Ty, Idx);
}

void GPUSanImpl::buildCallTreeInfo(Function &Fn, LocationInfoTy &LI) {
  if (Fn.hasFnAttribute("kernel"))
    return;
  SmallVector<CallBase *> Calls;
  for (auto &U : Fn.uses()) {
    auto *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB)
      continue;
    if (!CB->isCallee(&U))
      continue;
    Calls.push_back(CB);
  }
  if (Calls.size() == 1) {
    getSourceIndex(*Calls.back(), &LI);
    return;
  }
  LI.ParentIdx = -2;
  AmbiguousCalls.insert(Calls.begin(), Calls.end());
}

Value *GPUSanImpl::getPC(IRBuilder<> &IRB) {
  return IRB.CreateIntrinsic(Int64Ty, Intrinsic::amdgcn_s_getpc, {}, nullptr,
                             "PC");
}
Value *GPUSanImpl::getFunctionName(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  StringRef FnName = IRB.GetInsertPoint()->getFunction()->getName();
  if (DLoc && DLoc.get()) {
    StringRef SubprogramName = DLoc.get()->getSubprogramLinkageName();
    if (!SubprogramName.empty())
      FnName = SubprogramName;
  }
  StringRef Name = FnName.take_back(255);
  Value *&NameVal = GlobalStringMap[Name];
  if (!NameVal)
    NameVal = IRB.CreateAddrSpaceCast(
        IRB.CreateGlobalStringPtr(Name, "", DL.getDefaultGlobalsAddressSpace(),
                                  &M),
        PtrTy);
  return NameVal;
}
Value *GPUSanImpl::getFileName(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  if (!DLoc || DLoc->getFilename().empty())
    return ConstantPointerNull::get(PtrTy);
  StringRef Name = DLoc->getFilename().take_back(255);
  Value *&NameVal = GlobalStringMap[Name];
  if (!NameVal)
    NameVal = IRB.CreateAddrSpaceCast(
        IRB.CreateGlobalStringPtr(Name, "", DL.getDefaultGlobalsAddressSpace(),
                                  &M),
        PtrTy);
  return NameVal;
}
Value *GPUSanImpl::getLineNo(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  if (!DLoc)
    return Constant::getNullValue(Int64Ty);
  return ConstantInt::get(Int64Ty, DLoc.getLine());
}

void GPUSanImpl::getAllocationInfo(Function &Fn, PtrOrigin PO, Value &Object,
                                   Value *&Start, Value *&Length, Value *&Tag) {
  auto &It = AllocationInfoMap[{&Fn, &Object}];
  if (!It.Start) {
    auto *IP = dyn_cast<Instruction>(&Object);
    if (IP)
      IP = IP->getNextNode();
    else
      IP = &*Fn.getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
    IRBuilder<> IRB(IP);
    auto *CB = createCall(IRB, getAllocationInfoFn(PO),
                          {IRB.CreateAddrSpaceCast(&Object, getPtrTy(PO))});
    It.Start = IRB.CreateExtractValue(CB, {0});
    It.Length = IRB.CreateExtractValue(CB, {1});
    It.Tag = IRB.CreateExtractValue(CB, {2});
  }
  Start = It.Start;
  Length = It.Length;
  Tag = It.Tag;
}

PtrOrigin GPUSanImpl::getPtrOrigin(LoopInfo &LI, Value *Ptr,
                                   const Value **Object) {
  SmallVector<const Value *> Objects;
  getUnderlyingObjects(Ptr, Objects, &LI);
  if (Object && Objects.size() == 1)
    *Object = Objects.front();
  PtrOrigin PO = NONE;
  for (auto *Obj : Objects) {
    PtrOrigin ObjPO = HasAllocas ? UNKNOWN : GLOBAL;
    if (isa<AllocaInst>(Obj)) {
      ObjPO = LOCAL;
    } else if (isa<GlobalVariable>(Obj)) {
      ObjPO = GLOBAL;
    } else if (auto *II = dyn_cast<IntrinsicInst>(Obj)) {
      if (II->getIntrinsicID() == Intrinsic::amdgcn_implicitarg_ptr ||
          II->getIntrinsicID() == Intrinsic::amdgcn_dispatch_ptr)
        return SYSTEM;
    } else if (auto *CI = dyn_cast<CallInst>(Obj)) {
      if (auto *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with("ompx_")) {
          if (Callee->getName().ends_with("_global"))
            ObjPO = GLOBAL;
          else if (Callee->getName().ends_with("_local"))
            ObjPO = LOCAL;
        }
    } else if (auto *Arg = dyn_cast<Argument>(Obj)) {
      if (Arg->getParent()->hasFnAttribute("kernel"))
        ObjPO = GLOBAL;
    }
    if (PO == NONE || PO == ObjPO) {
      PO = ObjPO;
    } else {
      return UNKNOWN;
    }
  }
  return PO;
}

bool GPUSanImpl::instrumentGlobals() {
  Function *DtorFn =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::PrivateLinkage, "__san.dtor", &M);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> IRB(Entry);
  createCall(IRB, getLeakCheckFn());
  IRB.CreateRetVoid();
  appendToGlobalDtors(M, DtorFn, 0, nullptr);

  return true;

  Function *DTorFn;
  std::tie(DTorFn, std::ignore) = getOrCreateSanitizerCtorAndInitFunctions(
      M, "ompx.ctor", "ompx.init",
      /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) {
        appendToGlobalCtors(M, Ctor, 0, Ctor);
      });
  return true;
}

Value *GPUSanImpl::instrumentAllocation(Instruction &I, Value &Size,
                                        FunctionCallee Fn, PtrOrigin PO) {
  IRBuilder<> IRB(I.getNextNode());
  Value *PlainI = IRB.CreatePointerBitCastOrAddrSpaceCast(&I, getPtrTy(PO));
  static int AllocationId = 1;
  auto *CB =
      createCall(IRB, Fn,
                 {PlainI, &Size, ConstantInt::get(Int64Ty, AllocationId++),
                  getSourceIndex(I)},
                 I.getName() + ".san");
  SmallVector<LifetimeIntrinsic *> Lifetimes;
  I.replaceUsesWithIf(
      IRB.CreatePointerBitCastOrAddrSpaceCast(CB, I.getType()), [&](Use &U) {
        if (auto *LT = dyn_cast<LifetimeIntrinsic>(U.getUser())) {
          Lifetimes.push_back(LT);
          return false;
        }
        return U.getUser() != PlainI && U.getUser() != CB;
      });
  if (Lifetimes.empty())
    return CB;

  CB->setArgOperand(1, ConstantInt::get(Int64Ty, 0));
  for (auto *LT : Lifetimes) {
    if (LT->getIntrinsicID() == Intrinsic::lifetime_start) {
      IRB.SetInsertPoint(LT);
      createCall(IRB, getLifetimeStart(), {CB, LT->getArgOperand(0)});
    } else {
      IRB.SetInsertPoint(LT);
      createCall(IRB, getLifetimeEnd(), {CB, LT->getArgOperand(0)});
    }
  }
  return CB;
}

Value *GPUSanImpl::instrumentAllocaInst(LoopInfo &LI, AllocaInst &AI) {
  auto SizeOrNone = AI.getAllocationSize(DL);
  if (!SizeOrNone)
    llvm_unreachable("TODO");
  Value *Size = ConstantInt::get(Int64Ty, *SizeOrNone);
  return instrumentAllocation(AI, *Size, getNewFn(LOCAL), LOCAL);
}

void GPUSanImpl::instrumentAccess(LoopInfo &LI, Instruction &I, int PtrIdx,
                                  Type &AccessTy, bool IsRead) {
  Value *PtrOp = I.getOperand(PtrIdx);
  const Value *Object = nullptr;
  PtrOrigin PO = getPtrOrigin(LI, PtrOp, &Object);
  if (PO > GLOBAL)
    return;

  Value *Start = nullptr;
  Value *Length = nullptr;
  Value *Tag = nullptr;
  if (PO != UNKNOWN && Object)
    getAllocationInfo(*I.getFunction(), PO, *const_cast<Value *>(Object), Start,
                      Length, Tag);

  if (Loop *L = LI.getLoopFor(I.getParent())) {
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(*I.getFunction());
    const auto &LD = SE.getLoopDisposition(SE.getSCEVAtScope(PtrOp, L), L);
    LD->
  }

  static int32_t ReadAccessId = -1;
  static int32_t WriteAccessId = 1;
  const int32_t &AccessId = IsRead ? ReadAccessId-- : WriteAccessId++;

  auto TySize = DL.getTypeStoreSize(&AccessTy);
  assert(!TySize.isScalable());
  Value *Size = ConstantInt::get(Int64Ty, TySize.getFixedValue());
  IRBuilder<> IRB(&I);
  Value *PlainPtrOp =
      IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, getPtrTy(PO));
  CallInst *CB;
  if (Start) {
    CB = createCall(IRB, getCheckWithBaseFn(PO),
                    {PlainPtrOp, Start, Length, Tag, Size,
                     ConstantInt::get(Int64Ty, AccessId), getSourceIndex(I)},
                    I.getName() + ".san");
  } else {
    CB = createCall(IRB, getCheckFn(PO),
                    {PlainPtrOp, Size, ConstantInt::get(Int64Ty, AccessId),
                     getSourceIndex(I)},
                    I.getName() + ".san");
  }
  I.setOperand(PtrIdx,
               IRB.CreatePointerBitCastOrAddrSpaceCast(CB, PtrOp->getType()));
}

void GPUSanImpl::instrumentLoadInst(LoopInfo &LI, LoadInst &LoadI) {
  instrumentAccess(LI, LoadI, LoadInst::getPointerOperandIndex(),
                   *LoadI.getType(),
                   /*IsRead=*/true);
}

void GPUSanImpl::instrumentStoreInst(LoopInfo &LI, StoreInst &StoreI) {
  instrumentAccess(LI, StoreI, StoreInst::getPointerOperandIndex(),
                   *StoreI.getValueOperand()->getType(), /*IsRead=*/false);
}

void GPUSanImpl::instrumentGEPInst(LoopInfo &LI, GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getPointerOperand();
  PtrOrigin PO = getPtrOrigin(LI, PtrOp);
  if (PO > GLOBAL)
    return;

  GEP.setOperand(GetElementPtrInst::getPointerOperandIndex(),
                 Constant::getNullValue(PtrOp->getType()));
  IRBuilder<> IRB(GEP.getNextNode());
  Value *PlainPtrOp =
      IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, getPtrTy(PO));
  auto *CB = createCall(IRB, getGEPFn(PO),
                        {PlainPtrOp, UndefValue::get(Int64Ty), getPC(IRB)},
                        GEP.getName() + ".san");
  GEP.replaceAllUsesWith(
      IRB.CreatePointerBitCastOrAddrSpaceCast(CB, GEP.getType()));
  Value *Offset =
      new PtrToIntInst(&GEP, Int64Ty, GEP.getName() + ".san.offset", CB);
  CB->setArgOperand(1, Offset);
}

bool GPUSanImpl::instrumentCallInst(LoopInfo &LI, CallInst &CI) {
  bool Changed = false;
  if (isa<LifetimeIntrinsic>(CI))
    return Changed;
  if (auto *Fn = CI.getCalledFunction()) {
    if (Fn->getName().starts_with("__kmpc_target_init"))
      return Changed;
    if ((Fn->isDeclaration() || Fn->getName().starts_with("__kmpc") ||
         Fn->getName().starts_with("rpc_")) &&
        !Fn->getName().starts_with("ompx")) {
      IRBuilder<> IRB(&CI);
      for (int I = 0, E = CI.arg_size(); I != E; ++I) {
        Value *Op = CI.getArgOperand(I);
        if (!Op->getType()->isPointerTy())
          continue;
        PtrOrigin PO = getPtrOrigin(LI, Op);
        if (PO > GLOBAL)
          continue;
        Value *PlainOp =
            IRB.CreatePointerBitCastOrAddrSpaceCast(Op, getPtrTy(PO));
        auto *CB = createCall(IRB, getUnpackFn(PO), {PlainOp, getPC(IRB)},
                              Op->getName() + ".unpack");
        CI.setArgOperand(
            I, IRB.CreatePointerBitCastOrAddrSpaceCast(CB, Op->getType()));
        Changed = true;
      }
    }
  }
  return Changed;
}

bool GPUSanImpl::instrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;

  bool Changed = false;
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(Fn);
  SmallVector<std::pair<AllocaInst *, Value *>> Allocas;
  SmallVector<ReturnInst *> Returns;
  SmallVector<LoadInst *> Loads;
  SmallVector<StoreInst *> Stores;
  SmallVector<CallInst *> Calls;
  SmallVector<GetElementPtrInst *> GEPs;

  for (auto &I : instructions(Fn)) {
    switch (I.getOpcode()) {
    case Instruction::Alloca: {
      AllocaInst &AI = cast<AllocaInst>(I);
      Allocas.push_back({&AI, nullptr});
      Changed = true;
      break;
    }
    case Instruction::Load:
      Loads.push_back(&cast<LoadInst>(I));
      Changed = true;
      break;
    case Instruction::Store:
      Stores.push_back(&cast<StoreInst>(I));
      Changed = true;
      break;
    case Instruction::GetElementPtr:
      GEPs.push_back(&cast<GetElementPtrInst>(I));
      Changed = true;
      break;
    case Instruction::Call: {
      auto &CI = cast<CallInst>(I);
      Calls.push_back(&CI);
      if (CI.isIndirectCall())
        AmbiguousCalls.insert(&CI);
      break;
    }
    case Instruction::Ret:
      Returns.push_back(&cast<ReturnInst>(I));
      break;
    default:
      break;
    }
  }

  for (auto *Load : Loads)
    instrumentLoadInst(LI, *Load);
  for (auto *Store : Stores)
    instrumentStoreInst(LI, *Store);
  for (auto *GEP : GEPs)
    instrumentGEPInst(LI, *GEP);
  for (auto *Call : Calls)
    Changed |= instrumentCallInst(LI, *Call);
  for (auto &It : Allocas)
    It.second = instrumentAllocaInst(LI, *It.first);

  instrumentReturns(Allocas, Returns);

  return Changed;
}

void GPUSanImpl::instrumentReturns(
    SmallVectorImpl<std::pair<AllocaInst *, Value *>> &Allocas,
    SmallVectorImpl<ReturnInst *> &Returns) {
  if (Allocas.empty())
    return;
  for (auto *RI : Returns) {
    IRBuilder<> IRB(RI);
    createCall(IRB, getFreeNLocalFn(),
               {ConstantInt::get(Int32Ty, Allocas.size())});
  }
}

bool GPUSanImpl::instrument() {
  bool Changed = instrumentGlobals();
  HasAllocas = [&]() {
    for (Function &Fn : M)
      for (auto &I : instructions(Fn))
        if (isa<AllocaInst>(I))
          return true;
    return false;
  }();

  SmallVector<Function *> Kernels;
  for (Function &Fn : M) {
    if (Fn.hasFnAttribute("kernel"))
      Kernels.push_back(&Fn);
    if (!Fn.getName().contains("ompx") && !Fn.getName().contains("__kmpc") &&
        !Fn.getName().starts_with("rpc_")) {
      if (!Fn.hasFnAttribute(Attribute::DisableSanitizerInstrumentation)) {
        Changed |= instrumentFunction(Fn);
      } else if (!Fn.isDeclaration() &&
                 Fn.getName().contains("SanitizerTrapInfoTy")) {
      }
    }
  }

  SmallVector<CallBase *> AmbiguousCallsOrdered;
  SmallVector<Constant *> AmbiguousCallsMapping;
  if (LocationMap.empty())
    AmbiguousCalls.clear();
  for (size_t I = 0; I < AmbiguousCalls.size(); ++I) {
    CallBase &CB = *AmbiguousCalls[I];
    AmbiguousCallsOrdered.push_back(&CB);
    AmbiguousCallsMapping.push_back(getSourceIndex(CB));
  }

  uint64_t AmbiguousCallsBitWidth =
      llvm::Log2_64_Ceil(AmbiguousCalls.size() + 1);

  new GlobalVariable(M, Int64Ty, /*isConstant=*/true,
                     GlobalValue::ExternalLinkage,
                     ConstantInt::get(Int64Ty, AmbiguousCallsBitWidth),
                     "__san.num_ambiguous_calls", nullptr,
                     GlobalValue::ThreadLocalMode::NotThreadLocal, 1);

  if (size_t NumAmbiguousCalls = AmbiguousCalls.size()) {
    {
      auto *ArrayTy = ArrayType::get(Int64Ty, NumAmbiguousCalls);
      auto *GV = new GlobalVariable(
          M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
          ConstantArray::get(ArrayTy, AmbiguousCallsMapping),
          "__san.ambiguous_calls_mapping", nullptr,
          GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
      GV->setVisibility(GlobalValue::ProtectedVisibility);
    }

    auto *ArrayTy = ArrayType::get(Int64Ty, 1024);
    LocationsArray = new GlobalVariable(
        M, ArrayTy, /*isConstant=*/false, GlobalValue::PrivateLinkage,
        UndefValue::get(ArrayTy), "__san.calls", nullptr,
        GlobalValue::ThreadLocalMode::NotThreadLocal, 3);

    auto *OldFn = M.getFunction("__san_get_location_value");
    if (OldFn)
      OldFn->setName("");
    Function *LocationGetter = Function::Create(
        FunctionType::get(Int64Ty, false), GlobalValue::ExternalLinkage,
        "__san_get_location_value", M);
    if (OldFn) {
      OldFn->replaceAllUsesWith(LocationGetter);
      OldFn->eraseFromParent();
    }
    auto *EntryBB = BasicBlock::Create(Ctx, "entry", LocationGetter);
    IRBuilder<> IRB(EntryBB);
    Value *Idx = createCall(IRB, getThreadIdFn(), {}, "san.gtid");
    Value *Ptr = IRB.CreateGEP(Int64Ty, LocationsArray, {Idx});
    auto *LocationValue = IRB.CreateLoad(Int64Ty, Ptr);
    IRB.CreateRet(LocationValue);
  }

  Function *InitSharedFn =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::PrivateLinkage, "__san.init_shared", &M);
  auto *EntryBB = BasicBlock::Create(Ctx, "entry", InitSharedFn);
  IRBuilder<> IRB(EntryBB);
  if (!AmbiguousCalls.empty()) {
    Value *Idx = createCall(IRB, getThreadIdFn(), {}, "san.gtid");
    Value *Ptr = IRB.CreateGEP(Int64Ty, LocationsArray, {Idx});
    IRB.CreateStore(ConstantInt::get(Int64Ty, 0), Ptr);
  }
  IRB.CreateRetVoid();

  for (auto *KernelFn : Kernels) {
    IRBuilder<> IRB(&*KernelFn->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    createCall(IRB, InitSharedFn, {});
  }

  for (const auto &It : llvm::enumerate(AmbiguousCallsOrdered)) {
    IRBuilder<> IRB(It.value());
    Value *Idx = createCall(IRB, getThreadIdFn(), {}, "san.gtid");
    Value *Ptr = IRB.CreateGEP(Int64Ty, LocationsArray, {Idx});
    Value *OldVal = IRB.CreateLoad(Int64Ty, Ptr);
    Value *OldValShifted = IRB.CreateShl(
        OldVal, ConstantInt::get(Int64Ty, AmbiguousCallsBitWidth));
    Value *NewVal = IRB.CreateBinOp(Instruction::Or, OldValShifted,
                                    ConstantInt::get(Int64Ty, It.index() + 1));
    IRB.CreateStore(NewVal, Ptr);
    IRB.SetInsertPoint(It.value()->getNextNode());
    IRB.CreateStore(OldVal, Ptr);
  }

  auto *NamesTy = ArrayType::get(Int8Ty, ConcatenatedString.size() + 1);
  auto *Names = new GlobalVariable(
      M, NamesTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantDataArray::getString(Ctx, ConcatenatedString),
      "__san.location_names", nullptr,
      GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
  Names->setVisibility(GlobalValue::ProtectedVisibility);

  auto *ArrayTy = ArrayType::get(Int64Ty, LocationEncoding.size());
  auto *GV = new GlobalVariable(
      M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(ArrayTy, LocationEncoding), "__san.locations", nullptr,
      GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
  GV->setVisibility(GlobalValue::ProtectedVisibility);

  for (auto *CI : Calls) {
    InlineFunctionInfo IFI;
    if (InlineFunction(*CI, IFI).isSuccess())
      Changed = true;
  }

  return Changed;
}

PreservedAnalyses GPUSanPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  GPUSanImpl Lowerer(M, FAM);
  if (!Lowerer.instrument())
    return PreservedAnalyses::all();
  LLVM_DEBUG(M.dump());
  return PreservedAnalyses::none();
}
