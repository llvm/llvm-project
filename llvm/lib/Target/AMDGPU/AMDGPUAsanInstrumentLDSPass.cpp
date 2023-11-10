#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <iostream>
#define DEBUG_TYPE "amdgpu-asan-instrument-lds"

using namespace llvm;
using DomTreeCallback = function_ref<DominatorTree *(Function &F)>;

namespace {
// TODO: Just for testing purpose. Will be removed.
cl::opt<bool> ReplaceLDSAndInstrument(
    "amdgpu-replace-lds-and-instrument",
    cl::desc("Replace LDS accesses with malloc and don't do asan instrumentation."),
    cl::init(true), cl::Hidden);

const char kAMDGPUBallotName[] = "llvm.amdgcn.ballot.i64";
const char kAMDGPUUnreachableName[] = "llvm.amdgcn.unreachable";
static const uint64_t kSmallX86_64ShadowOffsetBase = 0x7FFFFFFF;
static const uint64_t kSmallX86_64ShadowOffsetAlignMask = ~0xFFFULL;
const bool Recover = true;
const uint32_t AsanMappingScale = 3;
const uint32_t AsanMappingOffset =
    (kSmallX86_64ShadowOffsetBase &
     (kSmallX86_64ShadowOffsetAlignMask << AsanMappingScale));

class AMDGPUAsanInstrumentLDS : public ModulePass {

public:
  static char ID;
  AMDGPUAsanInstrumentLDS() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequiredID(AMDGPULowerModuleLDSLegacyPassID);
  }
};
} // namespace

INITIALIZE_PASS(AMDGPUAsanInstrumentLDS, "amdgpu-asan-instrument-lds",
                "AMDGPU AddressSanitizer instrument LDS", false, false)

char AMDGPUAsanInstrumentLDS::ID = 0;

static uint64_t getRedzoneSizeForGlobal(uint64_t SizeInBytes) {
  constexpr uint64_t kMaxRZ = 1 << 18;
  // TODO: get scale from asan-mapping-scale
  const int MappingScale = AsanMappingScale;
  const uint64_t MinRZ = std::max(32U, 1U << MappingScale);
  ;

  uint64_t RZ = 0;
  if (SizeInBytes <= MinRZ / 2) {
    // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
    // at least 32 bytes, optimize when SizeInBytes is less than or equal to
    // half of MinRZ.
    RZ = MinRZ - SizeInBytes;
  } else {
    // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
    RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);

    // Round up to multiple of MinRZ.
    if (SizeInBytes % MinRZ)
      RZ += MinRZ - (SizeInBytes % MinRZ);
  }

  assert((RZ + SizeInBytes) % MinRZ == 0);

  return RZ;
}

static Instruction *genAMDGPUReportBlock(Module &M, IRBuilder<> &IRB,
                                         Value *Cond, bool Recover) {
  Value *ReportCond = Cond;
  if (!Recover) {
    auto Ballot = M.getOrInsertFunction(kAMDGPUBallotName, IRB.getInt64Ty(),
                                        IRB.getInt1Ty());
    ReportCond = IRB.CreateIsNotNull(IRB.CreateCall(Ballot, {Cond}));
  }

  auto *Trm = SplitBlockAndInsertIfThen(
      ReportCond, &*IRB.GetInsertPoint(), false,
      MDBuilder(M.getContext()).createBranchWeights(1, 100000));
  Trm->getParent()->setName("asan.report");

  if (Recover)
    return Trm;

  Trm = SplitBlockAndInsertIfThen(Cond, Trm, false);
  IRB.SetInsertPoint(Trm);
  return IRB.CreateCall(
      M.getOrInsertFunction(kAMDGPUUnreachableName, IRB.getVoidTy()), {});
}

static Value *createSlowPathCmp(Module &M, IRBuilder<> &IRB, Value *AddrLong,
                                Value *ShadowValue, uint32_t TypeStoreSize) {

  unsigned int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  size_t Granularity = static_cast<size_t>(1) << AsanMappingScale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte =
      IRB.CreateAnd(AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeStoreSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeStoreSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte =
      IRB.CreateIntCast(LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

static size_t TypeStoreSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = llvm::countr_zero(TypeSize / 8);
  return Res;
}

static Instruction *generateCrashCode(Module &M, IRBuilder<> &IRB,
                                      Instruction *InsertBefore, Value *Addr,
                                      bool IsWrite, size_t AccessSizeIndex,
                                      Value *SizeArgument) {
  IRB.SetInsertPoint(InsertBefore);
  CallInst *Call = nullptr;
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  const char kAsanReportErrorTemplate[] = "__asan_report_";
  const std::string TypeStr = IsWrite ? "store" : "load";
  const std::string EndingStr = Recover ? "_noabort" : "";
  SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
  AttributeList AL2;
  FunctionCallee AsanErrorCallbackSized = M.getOrInsertFunction(
      kAsanReportErrorTemplate + TypeStr + "_n" + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);
  const std::string Suffix = TypeStr + llvm::itostr(1ULL << AccessSizeIndex);
  SmallVector<Type *, 2> Args1{1, IntptrTy};
  AttributeList AL1;
  FunctionCallee AsanErrorCallback = M.getOrInsertFunction(
      kAsanReportErrorTemplate + Suffix + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);
  if (SizeArgument) {
    Call = IRB.CreateCall(AsanErrorCallbackSized, {Addr, SizeArgument});
  } else {
    Call = IRB.CreateCall(AsanErrorCallback, Addr);
  }

  Call->setCannotMerge();
  return Call;
}

static Value *memToShadow(Module &M, Value *Shadow, IRBuilder<> &IRB) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, AsanMappingScale);
  if (AsanMappingOffset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  Value *ShadowBase = ConstantInt::get(IntptrTy, AsanMappingOffset);
  return IRB.CreateAdd(Shadow, ShadowBase);
}

static void InstrumentAddress(Module &M, IRBuilder<> &IRB, Instruction *OrigIns,
                              Instruction *InsertBefore, Value *Addr,
                              MaybeAlign Alignment, uint32_t TypeStoreSize,
                              bool IsWrite, Value *SizeArgument,
                              bool UseCalls) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRB.SetInsertPoint(InsertBefore);
  size_t AccessSizeIndex = TypeStoreSizeToSizeIndex(TypeStoreSize);
  Type *ShadowTy = IntegerType::get(
      M.getContext(), std::max(8U, TypeStoreSize >> AsanMappingScale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *ShadowPtr = memToShadow(M, AddrLong, IRB);
  const uint64_t ShadowAlign =
      std::max<uint64_t>(Alignment.valueOrOne().value() >> AsanMappingScale, 1);
  Value *ShadowValue = IRB.CreateAlignedLoad(
      ShadowTy, IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy), Align(ShadowAlign));
  Value *Cmp = IRB.CreateIsNotNull(ShadowValue);
  size_t Granularity = 1ULL << AsanMappingScale;
  Instruction *CrashTerm = nullptr;
  auto *Cmp2 = createSlowPathCmp(M, IRB, AddrLong, ShadowValue, TypeStoreSize);
  Cmp = IRB.CreateAnd(Cmp, Cmp2);
  CrashTerm = genAMDGPUReportBlock(M, IRB, Cmp, Recover);
  Instruction *Crash = generateCrashCode(M, IRB, CrashTerm, AddrLong, IsWrite,
                                         AccessSizeIndex, SizeArgument);
  return;
}

static GlobalVariable *ReplaceLDSWithMalloc(IRBuilder<> &IRB, Module &M,
                                            Function *Func,
                                            GlobalVariable *LoweredLDSGlobal,
                                            DomTreeUpdater &DTU) {
  // TODO
  // Do single malloc for all globals. store offsets of GV into malloc
  // Store malloc pointer to LDS
  // Replace lds accesses with lds malloc ptr + offsets
  StructType *LoweredLDSGlobalTy =
      dyn_cast<StructType>(LoweredLDSGlobal->getValueType());
  if (!LoweredLDSGlobalTy)
    return nullptr;

  auto &Ctx = M.getContext();
  // Store previous entry block
  auto *PrevEntryBlock = &Func->getEntryBlock();

  // Create malloc block
  auto *MallocBlock = BasicBlock::Create(Ctx, "Malloc", Func, PrevEntryBlock);

  // Create WIdBlock block
  auto *WIdBlock = BasicBlock::Create(Ctx, "WId", Func, MallocBlock);
  IRB.SetInsertPoint(WIdBlock, WIdBlock->begin());
  auto *const WIdx =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x, {}, {});
  auto *const WIdy =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_y, {}, {});
  auto *const WIdz =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_z, {}, {});
  auto *const XYOr = IRB.CreateOr(WIdx, WIdy);
  auto *const XYZOr = IRB.CreateOr(XYOr, WIdz);
  auto *const WIdzCond = IRB.CreateICmpEQ(XYZOr, IRB.getInt32(0));
  IRB.CreateCondBr(WIdzCond, MallocBlock, PrevEntryBlock);

  // Malloc block
  IRB.SetInsertPoint(MallocBlock, MallocBlock->begin());
  const char kAsanMallocImplName[] = "malloc";
  FunctionCallee AsanAMDGPUMallocReturn = M.getOrInsertFunction(
      kAsanMallocImplName,
      FunctionType::get(IRB.getPtrTy(), {IRB.getInt64Ty()}, false));

  auto &DL = M.getDataLayout();
  uint64_t MallocOffset = 0;
  uint64_t MemberOffset = 0;
  DenseMap<uint64_t, uint64_t> MemberOffsetToMallocOffsetMap;
  for (auto I : llvm::enumerate(LoweredLDSGlobalTy->elements())) {
    Type *Ty = I.value();
    MemberOffsetToMallocOffsetMap[MemberOffset] = MallocOffset;
    const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
    const uint64_t RightRedzoneSize = getRedzoneSizeForGlobal(SizeInBytes);
    MallocOffset += SizeInBytes + RightRedzoneSize;
    ++MemberOffset;
  }

  ConstantInt *MallocSizeArg =
      ConstantInt::get(Type::getInt64Ty(Ctx), MallocOffset);
  Value *MCI = IRB.CreateCall(AsanAMDGPUMallocReturn, {MallocSizeArg});
  // create new global pointer variable
  GlobalVariable *NewGlobal = new GlobalVariable(
      M, IRB.getPtrTy(), false, GlobalValue::InternalLinkage,
      PoisonValue::get(IRB.getPtrTy()),
      Twine("llvm.amdgcn.asan." + Func->getName() + ".lds"), nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
  // create load of malloc to new global
  IRB.CreateStore(MCI, NewGlobal);

  // Replace lds accesses with malloc ptr + offsets
  for (Use &U : make_early_inc_range(LoweredLDSGlobal->uses())) {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(U.getUser())) {
      Instruction *UserI = dyn_cast<Instruction>(GEP->use_begin()->getUser());
      unsigned Indices = GEP->getNumIndices();
      MemberOffset =
          cast<ConstantInt>(GEP->getOperand(Indices))->getZExtValue();
      Type *Ty = LoweredLDSGlobalTy->elements()[MemberOffset];
      MallocOffset = MemberOffsetToMallocOffsetMap[MemberOffset];
      ConstantInt *OffsetConst =
          ConstantInt::get(Type::getInt64Ty(Ctx), MallocOffset);
      Constant *AddrPlusOffset =
          ConstantExpr::getGetElementPtr(Ty, NewGlobal, {OffsetConst}, true);
      U.getUser()->replaceAllUsesWith(AddrPlusOffset);
      continue;
    } else {
      MemberOffset = 0;
      Type *Ty = LoweredLDSGlobalTy->elements()[MemberOffset];
      MallocOffset = MemberOffsetToMallocOffsetMap[MemberOffset];
      ConstantInt *OffsetConst =
          ConstantInt::get(Type::getInt64Ty(Ctx), MallocOffset);
      Constant *AddrPlusOffset =
          ConstantExpr::getGetElementPtr(Ty, NewGlobal, {OffsetConst}, true);
      U.set(AddrPlusOffset);
    }
  }

  // Add instrumented globals to llvm.compiler.used list to avoid LTO from
  // ConstantMerge'ing them.
  appendToCompilerUsed(M, {NewGlobal});

  // Create branch to PrevEntryBlock
  IRB.CreateBr(PrevEntryBlock);

  // Create wave-group barrier at the starting of Previous entry block
  Type *Int1Ty = IRB.getInt1Ty();
  IRB.SetInsertPoint(PrevEntryBlock, PrevEntryBlock->begin());
  auto *XYZCondPhi = IRB.CreatePHI(Int1Ty, 2, "xyzCond");
  XYZCondPhi->addIncoming(IRB.getInt1(0), WIdBlock);
  XYZCondPhi->addIncoming(IRB.getInt1(1), MallocBlock);

  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  auto *CondFreeBlock = BasicBlock::Create(Ctx, "CondFree", Func);
  auto *FreeBlock = BasicBlock::Create(Ctx, "Free", Func);
  auto *EndBlock = BasicBlock::Create(Ctx, "End", Func);
  DenseMap<BasicBlock *, Value *> BBToRetValMap;
  for (BasicBlock &BB : *Func) {
    if (!BB.empty()) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&BB.back())) {
        BasicBlock *Block = &BB;
        Value *Val = RI->getReturnValue();
        BBToRetValMap[Block] = Val;
        RI->eraseFromParent();
        IRB.SetInsertPoint(&BB, BB.end());
        IRB.CreateBr(CondFreeBlock);
      }
    }
  }

  // assert(BBToRetValMap.empty() && "Function has no return");
  IRB.SetInsertPoint(CondFreeBlock, CondFreeBlock->begin());

  const uint64_t Size = BBToRetValMap.size();
  auto First = BBToRetValMap.begin();
  auto Pair = *First;
  Value *Val = Pair.second;
  Type *FPhiTy = Val ? Pair.second->getType() : IRB.getVoidTy();
  auto *CFPhi = Val ? IRB.CreatePHI(FPhiTy, Size) : nullptr;

  for (auto &Entry : BBToRetValMap) {
    BasicBlock *BB = Entry.first;
    Value *Val = Entry.second;
    ;
    if (Val)
      CFPhi->addIncoming(Val, BB);
  }
  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  IRB.CreateCondBr(XYZCondPhi, FreeBlock, EndBlock);

  // Free Block
  IRB.SetInsertPoint(FreeBlock, FreeBlock->begin());
  const char kAsanFreeImplName[] = "free";
  FunctionCallee AsanAMDGPUFreeReturn = M.getOrInsertFunction(
      kAsanFreeImplName,
      FunctionType::get(IRB.getVoidTy(), {IRB.getPtrTy()}, false));

  Value *MallocPtr = IRB.CreateLoad(IRB.getPtrTy(), NewGlobal);
  IRB.CreateCall(AsanAMDGPUFreeReturn, {MallocPtr});

  IRB.CreateBr(EndBlock);

  // End Block
  IRB.SetInsertPoint(EndBlock, EndBlock->begin());
  if (CFPhi)
    IRB.CreateRet(CFPhi);
  else
    IRB.CreateRetVoid();

  DTU.applyUpdates({{DominatorTree::Insert, WIdBlock, MallocBlock},
                    {DominatorTree::Insert, MallocBlock, PrevEntryBlock},
                    {DominatorTree::Insert, PrevEntryBlock, CondFreeBlock},
                    {DominatorTree::Insert, CondFreeBlock, FreeBlock},
                    {DominatorTree::Insert, FreeBlock, EndBlock}});
  return NewGlobal;
}

static uint32_t getKernelLdsSizeAttributeAsInt(Function &F) {
  StringRef S = F.getFnAttribute("amdgpu-lds-size").getValueAsString();
  uint32_t LdsSize = 0;
  if (!S.empty())
    S.consumeInteger(0, LdsSize);
  return LdsSize;
}

static GlobalVariable *getKernelLDSGlobal(Module &M, Function &F) {
  SmallString<64> KernelLDSName("llvm.amdgcn.kernel.");
  KernelLDSName += F.getName();
  KernelLDSName += ".lds";
  return M.getNamedGlobal(KernelLDSName);
}

static bool AMDGPUAsanInstrumentLDSImpl(Module &M, DomTreeCallback DTCallback) {
  if (!AMDGPUTargetMachine::EnableLowerModuleLDS)
    return false;
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  SmallVector<GlobalVariable *, 16> NewLdsMallocGlobals;
  for (Function &F : M) {
    bool hasSanitizeAddrAttr = F.hasFnAttribute(Attribute::SanitizeAddress);
    GlobalVariable *GV = getKernelLDSGlobal(M, F);
    uint32_t StaticLdsSize = getKernelLdsSizeAttributeAsInt(F);
    if (hasSanitizeAddrAttr && (GV || (StaticLdsSize != 0))) {
      DomTreeUpdater DTU(DTCallback(F), DomTreeUpdater::UpdateStrategy::Lazy);
      GlobalVariable *NewGlobal = ReplaceLDSWithMalloc(IRB, M, &F, {GV}, DTU);
      if (NewGlobal)
        NewLdsMallocGlobals.push_back(NewGlobal);
    }
  }

  if (ReplaceLDSAndInstrument) {
    for (GlobalVariable *GV : NewLdsMallocGlobals) {
      // Iterate over users instructions of global
      for (Use &U : make_early_inc_range(GV->uses())) {
        if (GEPOperator *GEP = dyn_cast<GEPOperator>(U.getUser())) {
          Instruction *UserI = dyn_cast<Instruction>(GEP->use_begin()->getUser());
          bool IsStore = UserI ? isa<StoreInst>(UserI) : false;
          InstrumentAddress(M, IRB, UserI, UserI, U, {}, 8, IsStore, nullptr,
                            false);
        } else {
          Instruction *UserI = dyn_cast<Instruction>(U.getUser());
          if (!UserI)
            continue;
          bool IsStore = UserI ? isa<StoreInst>(UserI) : false;
          InstrumentAddress(M, IRB, UserI, UserI, U, {}, 8, IsStore, nullptr,
                            false);
        }
      }
    }
  }
  return false;
}

bool AMDGPUAsanInstrumentLDS::runOnModule(Module &M) {
  DominatorTreeWrapperPass *const DTW =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto DTCallback = [&DTW](Function &F) -> DominatorTree * {
    return DTW ? &DTW->getDomTree() : nullptr;
  };
  bool IsChanged = AMDGPUAsanInstrumentLDSImpl(M, DTCallback);
  return IsChanged;
}

ModulePass *llvm::createAMDGPUAsanInstrumentLDSPass() {
  return new AMDGPUAsanInstrumentLDS();
}

PreservedAnalyses AMDGPUAsanInstrumentLDSPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto DTCallback = [&FAM](Function &F) -> DominatorTree * {
    return &FAM.getResult<DominatorTreeAnalysis>(F);
  };
  bool IsChanged = AMDGPUAsanInstrumentLDSImpl(M, DTCallback);
  if (!IsChanged)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
