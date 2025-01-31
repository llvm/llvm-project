//===-- AMDGPUSwLowerLDS.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the local data store, LDS, uses in kernel and non-kernel
// functions in module to use dynamically allocated global memory.
// Packed LDS Layout is emulated in the global memory.
// The lowered memory instructions from LDS to global memory are then
// instrumented for address sanitizer, to catch addressing errors.
// This pass only work when address sanitizer has been enabled and has
// instrumented the IR. It identifies that IR has been instrumented using
// "nosanitize_address" module flag.
//
// Replacement of Kernel LDS accesses:
//    For a kernel, LDS access can be static or dynamic which are direct
//    (accessed within kernel) and indirect (accessed through non-kernels).
//    All these LDS accesses corresponding to kernel will be packed together,
//    where all static LDS accesses will be allocated first and then dynamic
//    LDS follows. The total size with alignment is calculated. A new LDS global
//    will be created for the kernel called "SW LDS" and it will have the
//    attribute "amdgpu-lds-size" attached with value of the size calculated.
//    All the LDS accesses in the module will be replaced by GEP with offset
//    into the "Sw LDS".
//    A new "llvm.amdgcn.<kernel>.dynlds" is created per kernel accessing
//    the dynamic LDS. This will be marked used by kernel and will have
//    MD_absolue_symbol metadata set to total static LDS size, Since dynamic
//    LDS allocation starts after all static LDS allocation.
//
//    A device global memory equal to the total LDS size will be allocated.
//    At the prologue of the kernel, a single work-item from the
//    work-group, does a "malloc" and stores the pointer of the
//    allocation in "SW LDS".
//
//    To store the offsets corresponding to all LDS accesses, another global
//    variable is created which will be called "SW LDS metadata" in this pass.
//    - SW LDS Global:
//        It is LDS global of ptr type with name
//        "llvm.amdgcn.sw.lds.<kernel-name>".
//    - Metadata Global:
//        It is of struct type, with n members. n equals the number of LDS
//        globals accessed by the kernel(direct and indirect). Each member of
//        struct is another struct of type {i32, i32, i32}. First member
//        corresponds to offset, second member corresponds to size of LDS global
//        being replaced and third represents the total aligned size. It will
//        have name "llvm.amdgcn.sw.lds.<kernel-name>.md". This global will have
//        an intializer with static LDS related offsets and sizes initialized.
//        But for dynamic LDS related entries, offsets will be intialized to
//        previous static LDS allocation end offset. Sizes for them will be zero
//        initially. These dynamic LDS offset and size values will be updated
//        within the kernel, since kernel can read the dynamic LDS size
//        allocation done at runtime with query to "hidden_dynamic_lds_size"
//        hidden kernel argument.
//
//    At the epilogue of kernel, allocated memory would be made free by the same
//    single work-item.
//
// Replacement of non-kernel LDS accesses:
//    Multiple kernels can access the same non-kernel function.
//    All the kernels accessing LDS through non-kernels are sorted and
//    assigned a kernel-id. All the LDS globals accessed by non-kernels
//    are sorted. This information is used to build two tables:
//    - Base table:
//        Base table will have single row, with elements of the row
//        placed as per kernel ID. Each element in the row corresponds
//        to ptr of "SW LDS" variable created for that kernel.
//    - Offset table:
//        Offset table will have multiple rows and columns.
//        Rows are assumed to be from 0 to (n-1). n is total number
//        of kernels accessing the LDS through non-kernels.
//        Each row will have m elements. m is the total number of
//        unique LDS globals accessed by all non-kernels.
//        Each element in the row correspond to the ptr of
//        the replacement of LDS global done by that particular kernel.
//    A LDS variable in non-kernel will be replaced based on the information
//    from base and offset tables. Based on kernel-id query, ptr of "SW
//    LDS" for that corresponding kernel is obtained from base table.
//    The Offset into the base "SW LDS" is obtained from
//    corresponding element in offset table. With this information, replacement
//    value is obtained.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUAsanInstrumentation.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>

#define DEBUG_TYPE "amdgpu-sw-lower-lds"
#define COV5_HIDDEN_DYN_LDS_SIZE_ARG 15

using namespace llvm;
using namespace AMDGPU;

namespace {

cl::opt<bool>
    AsanInstrumentLDS("amdgpu-asan-instrument-lds",
                      cl::desc("Run asan instrumentation on LDS instructions "
                               "lowered to global memory"),
                      cl::init(true), cl::Hidden);

using DomTreeCallback = function_ref<DominatorTree *(Function &F)>;

struct LDSAccessTypeInfo {
  SetVector<GlobalVariable *> StaticLDSGlobals;
  SetVector<GlobalVariable *> DynamicLDSGlobals;
};

// Struct to hold all the Metadata required for a kernel
// to replace a LDS global uses with corresponding offset
// in to device global memory.
struct KernelLDSParameters {
  GlobalVariable *SwLDS = nullptr;
  GlobalVariable *SwDynLDS = nullptr;
  GlobalVariable *SwLDSMetadata = nullptr;
  LDSAccessTypeInfo DirectAccess;
  LDSAccessTypeInfo IndirectAccess;
  DenseMap<GlobalVariable *, SmallVector<uint32_t, 3>>
      LDSToReplacementIndicesMap;
  uint32_t MallocSize = 0;
  uint32_t LDSSize = 0;
  SmallVector<std::pair<uint32_t, uint32_t>, 64> RedzoneOffsetAndSizeVector;
};

// Struct to store information for creation of offset table
// for all the non-kernel LDS accesses.
struct NonKernelLDSParameters {
  GlobalVariable *LDSBaseTable = nullptr;
  GlobalVariable *LDSOffsetTable = nullptr;
  SetVector<Function *> OrderedKernels;
  SetVector<GlobalVariable *> OrdereLDSGlobals;
};

struct AsanInstrumentInfo {
  int Scale = 0;
  uint32_t Offset = 0;
  SetVector<Instruction *> Instructions;
};

struct FunctionsAndLDSAccess {
  DenseMap<Function *, KernelLDSParameters> KernelToLDSParametersMap;
  SetVector<Function *> KernelsWithIndirectLDSAccess;
  SetVector<Function *> NonKernelsWithLDSArgument;
  SetVector<GlobalVariable *> AllNonKernelLDSAccess;
  FunctionVariableMap NonKernelToLDSAccessMap;
};

class AMDGPUSwLowerLDS {
public:
  AMDGPUSwLowerLDS(Module &Mod, const AMDGPUTargetMachine &TM,
                   DomTreeCallback Callback)
      : M(Mod), AMDGPUTM(TM), IRB(M.getContext()), DTCallback(Callback) {}
  bool run();
  void getUsesOfLDSByNonKernels();
  void getNonKernelsWithLDSArguments(const CallGraph &CG);
  SetVector<Function *>
  getOrderedIndirectLDSAccessingKernels(SetVector<Function *> &Kernels);
  SetVector<GlobalVariable *>
  getOrderedNonKernelAllLDSGlobals(SetVector<GlobalVariable *> &Variables);
  void buildSwLDSGlobal(Function *Func);
  void buildSwDynLDSGlobal(Function *Func);
  void populateSwMetadataGlobal(Function *Func);
  void populateSwLDSAttributeAndMetadata(Function *Func);
  void populateLDSToReplacementIndicesMap(Function *Func);
  void getLDSMemoryInstructions(Function *Func,
                                SetVector<Instruction *> &LDSInstructions);
  void replaceKernelLDSAccesses(Function *Func);
  Value *getTranslatedGlobalMemoryPtrOfLDS(Value *LoadMallocPtr, Value *LDSPtr);
  void translateLDSMemoryOperationsToGlobalMemory(
      Function *Func, Value *LoadMallocPtr,
      SetVector<Instruction *> &LDSInstructions);
  void poisonRedzones(Function *Func, Value *MallocPtr);
  void lowerKernelLDSAccesses(Function *Func, DomTreeUpdater &DTU);
  void buildNonKernelLDSOffsetTable(NonKernelLDSParameters &NKLDSParams);
  void buildNonKernelLDSBaseTable(NonKernelLDSParameters &NKLDSParams);
  Constant *
  getAddressesOfVariablesInKernel(Function *Func,
                                  SetVector<GlobalVariable *> &Variables);
  void lowerNonKernelLDSAccesses(Function *Func,
                                 SetVector<GlobalVariable *> &LDSGlobals,
                                 NonKernelLDSParameters &NKLDSParams);
  void
  updateMallocSizeForDynamicLDS(Function *Func, Value **CurrMallocSize,
                                Value *HiddenDynLDSSize,
                                SetVector<GlobalVariable *> &DynamicLDSGlobals);
  void initAsanInfo();

private:
  Module &M;
  const AMDGPUTargetMachine &AMDGPUTM;
  IRBuilder<> IRB;
  DomTreeCallback DTCallback;
  FunctionsAndLDSAccess FuncLDSAccessInfo;
  AsanInstrumentInfo AsanInfo;
};

template <typename T> SetVector<T> sortByName(std::vector<T> &&V) {
  // Sort the vector of globals or Functions based on their name.
  // Returns a SetVector of globals/Functions.
  sort(V, [](const auto *L, const auto *R) {
    return L->getName() < R->getName();
  });
  return {SetVector<T>(V.begin(), V.end())};
}

SetVector<GlobalVariable *> AMDGPUSwLowerLDS::getOrderedNonKernelAllLDSGlobals(
    SetVector<GlobalVariable *> &Variables) {
  // Sort all the non-kernel LDS accesses based on their name.
  return sortByName(
      std::vector<GlobalVariable *>(Variables.begin(), Variables.end()));
}

SetVector<Function *> AMDGPUSwLowerLDS::getOrderedIndirectLDSAccessingKernels(
    SetVector<Function *> &Kernels) {
  // Sort the non-kernels accessing LDS based on their name.
  // Also assign a kernel ID metadata based on the sorted order.
  LLVMContext &Ctx = M.getContext();
  if (Kernels.size() > UINT32_MAX) {
    report_fatal_error("Unimplemented SW LDS lowering for > 2**32 kernels");
  }
  SetVector<Function *> OrderedKernels =
      sortByName(std::vector<Function *>(Kernels.begin(), Kernels.end()));
  for (size_t i = 0; i < Kernels.size(); i++) {
    Metadata *AttrMDArgs[1] = {
        ConstantAsMetadata::get(IRB.getInt32(i)),
    };
    Function *Func = OrderedKernels[i];
    Func->setMetadata("llvm.amdgcn.lds.kernel.id",
                      MDNode::get(Ctx, AttrMDArgs));
  }
  return OrderedKernels;
}

void AMDGPUSwLowerLDS::getNonKernelsWithLDSArguments(const CallGraph &CG) {
  // Among the kernels accessing LDS, get list of
  // Non-kernels to which a call is made and a ptr
  // to addrspace(3) is passed as argument.
  for (auto &K : FuncLDSAccessInfo.KernelToLDSParametersMap) {
    Function *Func = K.first;
    const CallGraphNode *CGN = CG[Func];
    if (!CGN)
      continue;
    for (auto &I : *CGN) {
      CallGraphNode *CallerCGN = I.second;
      Function *CalledFunc = CallerCGN->getFunction();
      if (!CalledFunc || CalledFunc->isDeclaration())
        continue;
      if (AMDGPU::isKernelLDS(CalledFunc))
        continue;
      for (auto AI = CalledFunc->arg_begin(), E = CalledFunc->arg_end();
           AI != E; ++AI) {
        Type *ArgTy = (*AI).getType();
        if (!ArgTy->isPointerTy())
          continue;
        if (ArgTy->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
          continue;
        FuncLDSAccessInfo.NonKernelsWithLDSArgument.insert(CalledFunc);
        // Also add the Calling function to KernelsWithIndirectLDSAccess list
        // so that base table of LDS is generated.
        FuncLDSAccessInfo.KernelsWithIndirectLDSAccess.insert(Func);
      }
    }
  }
}

void AMDGPUSwLowerLDS::getUsesOfLDSByNonKernels() {
  for (GlobalVariable *GV : FuncLDSAccessInfo.AllNonKernelLDSAccess) {
    if (!AMDGPU::isLDSVariableToLower(*GV))
      continue;

    for (User *V : GV->users()) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        Function *F = I->getFunction();
        if (!isKernelLDS(F) && F->hasFnAttribute(Attribute::SanitizeAddress) &&
            !F->isDeclaration())
          FuncLDSAccessInfo.NonKernelToLDSAccessMap[F].insert(GV);
      }
    }
  }
}

static void recordLDSAbsoluteAddress(Module &M, GlobalVariable *GV,
                                     uint32_t Address) {
  // Write the specified address into metadata where it can be retrieved by
  // the assembler. Format is a half open range, [Address Address+1)
  LLVMContext &Ctx = M.getContext();
  auto *IntTy = M.getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LOCAL_ADDRESS);
  MDBuilder MDB(Ctx);
  MDNode *MetadataNode = MDB.createRange(ConstantInt::get(IntTy, Address),
                                         ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol, MetadataNode);
}

static void addLDSSizeAttribute(Function *Func, uint32_t Offset,
                                bool IsDynLDS) {
  if (Offset != 0) {
    std::string Buffer;
    raw_string_ostream SS{Buffer};
    SS << Offset;
    if (IsDynLDS)
      SS << "," << Offset;
    Func->addFnAttr("amdgpu-lds-size", Buffer);
  }
}

static void markUsedByKernel(Function *Func, GlobalVariable *SGV) {
  BasicBlock *Entry = &Func->getEntryBlock();
  IRBuilder<> Builder(Entry, Entry->getFirstNonPHIIt());

  Function *Decl = Intrinsic::getOrInsertDeclaration(Func->getParent(),
                                                     Intrinsic::donothing, {});

  Value *UseInstance[1] = {
      Builder.CreateConstInBoundsGEP1_32(SGV->getValueType(), SGV, 0)};

  Builder.CreateCall(Decl, {},
                     {OperandBundleDefT<Value *>("ExplicitUse", UseInstance)});
}

void AMDGPUSwLowerLDS::buildSwLDSGlobal(Function *Func) {
  // Create new LDS global required for each kernel to store
  // device global memory pointer.
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  // Create new global pointer variable
  LDSParams.SwLDS = new GlobalVariable(
      M, IRB.getPtrTy(), false, GlobalValue::InternalLinkage,
      PoisonValue::get(IRB.getPtrTy()), "llvm.amdgcn.sw.lds." + Func->getName(),
      nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  LDSParams.SwLDS->setSanitizerMetadata(MD);
}

void AMDGPUSwLowerLDS::buildSwDynLDSGlobal(Function *Func) {
  // Create new Dyn LDS global if kernel accesses dyn LDS.
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  if (LDSParams.DirectAccess.DynamicLDSGlobals.empty() &&
      LDSParams.IndirectAccess.DynamicLDSGlobals.empty())
    return;
  // Create new global pointer variable
  auto *emptyCharArray = ArrayType::get(IRB.getInt8Ty(), 0);
  LDSParams.SwDynLDS = new GlobalVariable(
      M, emptyCharArray, false, GlobalValue::ExternalLinkage, nullptr,
      "llvm.amdgcn." + Func->getName() + ".dynlds", nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
  markUsedByKernel(Func, LDSParams.SwDynLDS);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  LDSParams.SwDynLDS->setSanitizerMetadata(MD);
}

void AMDGPUSwLowerLDS::populateSwLDSAttributeAndMetadata(Function *Func) {
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  bool IsDynLDSUsed = LDSParams.SwDynLDS ? true : false;
  uint32_t Offset = LDSParams.LDSSize;
  recordLDSAbsoluteAddress(M, LDSParams.SwLDS, 0);
  addLDSSizeAttribute(Func, Offset, IsDynLDSUsed);
  if (LDSParams.SwDynLDS)
    recordLDSAbsoluteAddress(M, LDSParams.SwDynLDS, Offset);
}

void AMDGPUSwLowerLDS::populateSwMetadataGlobal(Function *Func) {
  // Create new metadata global for every kernel and initialize the
  // start offsets and sizes corresponding to each LDS accesses.
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  std::vector<Type *> Items;
  Type *Int32Ty = IRB.getInt32Ty();
  std::vector<Constant *> Initializers;
  Align MaxAlignment(1);
  auto UpdateMaxAlignment = [&MaxAlignment, &DL](GlobalVariable *GV) {
    Align GVAlign = AMDGPU::getAlign(DL, GV);
    MaxAlignment = std::max(MaxAlignment, GVAlign);
  };

  for (GlobalVariable *GV : LDSParams.DirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.DirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.IndirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.IndirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  //{StartOffset, AlignedSizeInBytes}
  SmallString<128> MDItemStr;
  raw_svector_ostream MDItemOS(MDItemStr);
  MDItemOS << "llvm.amdgcn.sw.lds." << Func->getName() << ".md.item";

  StructType *LDSItemTy =
      StructType::create(Ctx, {Int32Ty, Int32Ty, Int32Ty}, MDItemOS.str());
  uint32_t &MallocSize = LDSParams.MallocSize;
  SetVector<GlobalVariable *> UniqueLDSGlobals;
  int AsanScale = AsanInfo.Scale;
  auto buildInitializerForSwLDSMD =
      [&](SetVector<GlobalVariable *> &LDSGlobals) {
        for (auto &GV : LDSGlobals) {
          if (is_contained(UniqueLDSGlobals, GV))
            continue;
          UniqueLDSGlobals.insert(GV);

          Type *Ty = GV->getValueType();
          const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
          Items.push_back(LDSItemTy);
          Constant *ItemStartOffset = ConstantInt::get(Int32Ty, MallocSize);
          Constant *SizeInBytesConst = ConstantInt::get(Int32Ty, SizeInBytes);
          // Get redzone size corresponding a size.
          const uint64_t RightRedzoneSize =
              AMDGPU::getRedzoneSizeForGlobal(AsanScale, SizeInBytes);
          // Update MallocSize with current size and redzone size.
          MallocSize += SizeInBytes;
          if (!AMDGPU::isDynamicLDS(*GV))
            LDSParams.RedzoneOffsetAndSizeVector.emplace_back(MallocSize,
                                                              RightRedzoneSize);
          MallocSize += RightRedzoneSize;
          // Align current size plus redzone.
          uint64_t AlignedSize =
              alignTo(SizeInBytes + RightRedzoneSize, MaxAlignment);
          Constant *AlignedSizeInBytesConst =
              ConstantInt::get(Int32Ty, AlignedSize);
          // Align MallocSize
          MallocSize = alignTo(MallocSize, MaxAlignment);
          Constant *InitItem =
              ConstantStruct::get(LDSItemTy, {ItemStartOffset, SizeInBytesConst,
                                              AlignedSizeInBytesConst});
          Initializers.push_back(InitItem);
        }
      };
  SetVector<GlobalVariable *> SwLDSVector;
  SwLDSVector.insert(LDSParams.SwLDS);
  buildInitializerForSwLDSMD(SwLDSVector);
  buildInitializerForSwLDSMD(LDSParams.DirectAccess.StaticLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.IndirectAccess.StaticLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.DirectAccess.DynamicLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.IndirectAccess.DynamicLDSGlobals);

  // Update the LDS size used by the kernel.
  Type *Ty = LDSParams.SwLDS->getValueType();
  const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
  uint64_t AlignedSize = alignTo(SizeInBytes, MaxAlignment);
  LDSParams.LDSSize = AlignedSize;
  SmallString<128> MDTypeStr;
  raw_svector_ostream MDTypeOS(MDTypeStr);
  MDTypeOS << "llvm.amdgcn.sw.lds." << Func->getName() << ".md.type";
  StructType *MetadataStructType =
      StructType::create(Ctx, Items, MDTypeOS.str());
  SmallString<128> MDStr;
  raw_svector_ostream MDOS(MDStr);
  MDOS << "llvm.amdgcn.sw.lds." << Func->getName() << ".md";
  LDSParams.SwLDSMetadata = new GlobalVariable(
      M, MetadataStructType, false, GlobalValue::InternalLinkage,
      PoisonValue::get(MetadataStructType), MDOS.str(), nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::GLOBAL_ADDRESS, false);
  Constant *data = ConstantStruct::get(MetadataStructType, Initializers);
  LDSParams.SwLDSMetadata->setInitializer(data);
  assert(LDSParams.SwLDS);
  // Set the alignment to MaxAlignment for SwLDS.
  LDSParams.SwLDS->setAlignment(MaxAlignment);
  if (LDSParams.SwDynLDS)
    LDSParams.SwDynLDS->setAlignment(MaxAlignment);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  LDSParams.SwLDSMetadata->setSanitizerMetadata(MD);
}

void AMDGPUSwLowerLDS::populateLDSToReplacementIndicesMap(Function *Func) {
  // Fill the corresponding LDS replacement indices for each LDS access
  // related to this kernel.
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  SetVector<GlobalVariable *> UniqueLDSGlobals;
  auto PopulateIndices = [&](SetVector<GlobalVariable *> &LDSGlobals,
                             uint32_t &Idx) {
    for (auto &GV : LDSGlobals) {
      if (is_contained(UniqueLDSGlobals, GV))
        continue;
      UniqueLDSGlobals.insert(GV);
      LDSParams.LDSToReplacementIndicesMap[GV] = {0, Idx, 0};
      ++Idx;
    }
  };
  uint32_t Idx = 0;
  SetVector<GlobalVariable *> SwLDSVector;
  SwLDSVector.insert(LDSParams.SwLDS);
  PopulateIndices(SwLDSVector, Idx);
  PopulateIndices(LDSParams.DirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams.IndirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams.DirectAccess.DynamicLDSGlobals, Idx);
  PopulateIndices(LDSParams.IndirectAccess.DynamicLDSGlobals, Idx);
}

static void replacesUsesOfGlobalInFunction(Function *Func, GlobalVariable *GV,
                                           Value *Replacement) {
  // Replace all uses of LDS global in this Function with a Replacement.
  auto ReplaceUsesLambda = [Func](const Use &U) -> bool {
    auto *V = U.getUser();
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      auto *Func1 = Inst->getParent()->getParent();
      if (Func == Func1)
        return true;
    }
    return false;
  };
  GV->replaceUsesWithIf(Replacement, ReplaceUsesLambda);
}

void AMDGPUSwLowerLDS::replaceKernelLDSAccesses(Function *Func) {
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  GlobalVariable *SwLDS = LDSParams.SwLDS;
  assert(SwLDS);
  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDSMetadata);
  StructType *SwLDSMetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  Type *Int32Ty = IRB.getInt32Ty();
  auto &IndirectAccess = LDSParams.IndirectAccess;
  auto &DirectAccess = LDSParams.DirectAccess;
  // Replace all uses of LDS global in this Function with a Replacement.
  SetVector<GlobalVariable *> UniqueLDSGlobals;
  auto ReplaceLDSGlobalUses = [&](SetVector<GlobalVariable *> &LDSGlobals) {
    for (auto &GV : LDSGlobals) {
      // Do not generate instructions if LDS access is in non-kernel
      // i.e indirect-access.
      if ((IndirectAccess.StaticLDSGlobals.contains(GV) ||
           IndirectAccess.DynamicLDSGlobals.contains(GV)) &&
          (!DirectAccess.StaticLDSGlobals.contains(GV) &&
           !DirectAccess.DynamicLDSGlobals.contains(GV)))
        continue;
      if (is_contained(UniqueLDSGlobals, GV))
        continue;
      UniqueLDSGlobals.insert(GV);
      auto &Indices = LDSParams.LDSToReplacementIndicesMap[GV];
      assert(Indices.size() == 3);
      Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Indices[0]),
                            ConstantInt::get(Int32Ty, Indices[1]),
                            ConstantInt::get(Int32Ty, Indices[2])};
      Constant *GEP = ConstantExpr::getGetElementPtr(
          SwLDSMetadataStructType, SwLDSMetadata, GEPIdx, true);
      Value *Offset = IRB.CreateLoad(Int32Ty, GEP);
      Value *BasePlusOffset =
          IRB.CreateInBoundsGEP(IRB.getInt8Ty(), SwLDS, {Offset});
      LLVM_DEBUG(GV->printAsOperand(dbgs() << "Sw LDS Lowering, Replacing LDS ",
                                    false));
      replacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
    }
  };
  ReplaceLDSGlobalUses(DirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(IndirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(DirectAccess.DynamicLDSGlobals);
  ReplaceLDSGlobalUses(IndirectAccess.DynamicLDSGlobals);
}

void AMDGPUSwLowerLDS::updateMallocSizeForDynamicLDS(
    Function *Func, Value **CurrMallocSize, Value *HiddenDynLDSSize,
    SetVector<GlobalVariable *> &DynamicLDSGlobals) {
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  Type *Int32Ty = IRB.getInt32Ty();

  GlobalVariable *SwLDS = LDSParams.SwLDS;
  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDS && SwLDSMetadata);
  StructType *MetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  unsigned MaxAlignment = SwLDS->getAlignment();
  Value *MaxAlignValue = IRB.getInt32(MaxAlignment);
  Value *MaxAlignValueMinusOne = IRB.getInt32(MaxAlignment - 1);

  for (GlobalVariable *DynGV : DynamicLDSGlobals) {
    auto &Indices = LDSParams.LDSToReplacementIndicesMap[DynGV];
    // Update the Offset metadata.
    Constant *Index0 = ConstantInt::get(Int32Ty, 0);
    Constant *Index1 = ConstantInt::get(Int32Ty, Indices[1]);

    Constant *Index2Offset = ConstantInt::get(Int32Ty, 0);
    auto *GEPForOffset = IRB.CreateInBoundsGEP(
        MetadataStructType, SwLDSMetadata, {Index0, Index1, Index2Offset});

    IRB.CreateStore(*CurrMallocSize, GEPForOffset);
    // Update the size and Aligned Size metadata.
    Constant *Index2Size = ConstantInt::get(Int32Ty, 1);
    auto *GEPForSize = IRB.CreateInBoundsGEP(MetadataStructType, SwLDSMetadata,
                                             {Index0, Index1, Index2Size});

    Value *CurrDynLDSSize = IRB.CreateLoad(Int32Ty, HiddenDynLDSSize);
    IRB.CreateStore(CurrDynLDSSize, GEPForSize);
    Constant *Index2AlignedSize = ConstantInt::get(Int32Ty, 2);
    auto *GEPForAlignedSize = IRB.CreateInBoundsGEP(
        MetadataStructType, SwLDSMetadata, {Index0, Index1, Index2AlignedSize});

    Value *AlignedDynLDSSize =
        IRB.CreateAdd(CurrDynLDSSize, MaxAlignValueMinusOne);
    AlignedDynLDSSize = IRB.CreateUDiv(AlignedDynLDSSize, MaxAlignValue);
    AlignedDynLDSSize = IRB.CreateMul(AlignedDynLDSSize, MaxAlignValue);
    IRB.CreateStore(AlignedDynLDSSize, GEPForAlignedSize);

    // Update the Current Malloc Size
    *CurrMallocSize = IRB.CreateAdd(*CurrMallocSize, AlignedDynLDSSize);
  }
}

static DebugLoc getOrCreateDebugLoc(const Instruction *InsertBefore,
                                    DISubprogram *SP) {
  assert(InsertBefore);
  if (InsertBefore->getDebugLoc())
    return InsertBefore->getDebugLoc();
  if (SP)
    return DILocation::get(SP->getContext(), SP->getLine(), 1, SP);
  return DebugLoc();
}

void AMDGPUSwLowerLDS::getLDSMemoryInstructions(
    Function *Func, SetVector<Instruction *> &LDSInstructions) {
  for (BasicBlock &BB : *Func) {
    for (Instruction &Inst : BB) {
      if (LoadInst *LI = dyn_cast<LoadInst>(&Inst)) {
        if (LI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          LDSInstructions.insert(&Inst);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(&Inst)) {
        if (SI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          LDSInstructions.insert(&Inst);
      } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(&Inst)) {
        if (RMW->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          LDSInstructions.insert(&Inst);
      } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(&Inst)) {
        if (XCHG->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          LDSInstructions.insert(&Inst);
      } else if (AddrSpaceCastInst *ASC = dyn_cast<AddrSpaceCastInst>(&Inst)) {
        if (ASC->getSrcAddressSpace() == AMDGPUAS::LOCAL_ADDRESS &&
            ASC->getDestAddressSpace() == AMDGPUAS::FLAT_ADDRESS)
          LDSInstructions.insert(&Inst);
      } else
        continue;
    }
  }
}

Value *AMDGPUSwLowerLDS::getTranslatedGlobalMemoryPtrOfLDS(Value *LoadMallocPtr,
                                                           Value *LDSPtr) {
  assert(LDSPtr && "Invalid LDS pointer operand");
  Type *LDSPtrType = LDSPtr->getType();

  if (LDSPtrType->isVectorTy()) {
    // Handle vector of pointers
    VectorType *VecPtrTy = cast<VectorType>(LDSPtrType);
    ElementCount NumElements = VecPtrTy->getElementCount();
    Type *Int32VecTy = VectorType::get(IRB.getInt32Ty(), NumElements);
    Value *PtrToInt = IRB.CreatePtrToInt(LDSPtr, Int32VecTy);
    Type *GlobalPtrVecTy =
        VectorType::get(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), NumElements);
    Value *GlobalPtrVec = PoisonValue::get(GlobalPtrVecTy);
    for (uint64_t Index = 0; Index < NumElements.getKnownMinValue(); ++Index) {
      Value *ExtElem = IRB.CreateExtractElement(PtrToInt, Index);
      Value *Gep =
          IRB.CreateInBoundsGEP(IRB.getInt8Ty(), LoadMallocPtr, {ExtElem});
      GlobalPtrVec = IRB.CreateInsertElement(GlobalPtrVec, Gep, Index);
    }
    return GlobalPtrVec;
  }
  Value *PtrToInt = IRB.CreatePtrToInt(LDSPtr, IRB.getInt32Ty());
  return IRB.CreateInBoundsGEP(IRB.getInt8Ty(), LoadMallocPtr, {PtrToInt});
}

void AMDGPUSwLowerLDS::translateLDSMemoryOperationsToGlobalMemory(
    Function *Func, Value *LoadMallocPtr,
    SetVector<Instruction *> &LDSInstructions) {
  LLVM_DEBUG(dbgs() << "Translating LDS memory operations to global memory : "
                    << Func->getName());
  for (Instruction *Inst : LDSInstructions) {
    IRB.SetInsertPoint(Inst);
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
      Value *LIOperand = LI->getPointerOperand();
      Value *Replacement =
          getTranslatedGlobalMemoryPtrOfLDS(LoadMallocPtr, LIOperand);
      LoadInst *NewLI = IRB.CreateAlignedLoad(LI->getType(), Replacement,
                                              LI->getAlign(), LI->isVolatile());
      NewLI->setAtomic(LI->getOrdering(), LI->getSyncScopeID());
      AsanInfo.Instructions.insert(NewLI);
      LI->replaceAllUsesWith(NewLI);
      LI->eraseFromParent();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      Value *SIOperand = SI->getPointerOperand();
      Value *Replacement =
          getTranslatedGlobalMemoryPtrOfLDS(LoadMallocPtr, SIOperand);
      StoreInst *NewSI = IRB.CreateAlignedStore(
          SI->getValueOperand(), Replacement, SI->getAlign(), SI->isVolatile());
      NewSI->setAtomic(SI->getOrdering(), SI->getSyncScopeID());
      AsanInfo.Instructions.insert(NewSI);
      SI->replaceAllUsesWith(NewSI);
      SI->eraseFromParent();
    } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(Inst)) {
      Value *RMWPtrOperand = RMW->getPointerOperand();
      Value *RMWValOperand = RMW->getValOperand();
      Value *Replacement =
          getTranslatedGlobalMemoryPtrOfLDS(LoadMallocPtr, RMWPtrOperand);
      AtomicRMWInst *NewRMW = IRB.CreateAtomicRMW(
          RMW->getOperation(), Replacement, RMWValOperand, RMW->getAlign(),
          RMW->getOrdering(), RMW->getSyncScopeID());
      NewRMW->setVolatile(RMW->isVolatile());
      AsanInfo.Instructions.insert(NewRMW);
      RMW->replaceAllUsesWith(NewRMW);
      RMW->eraseFromParent();
    } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(Inst)) {
      Value *XCHGPtrOperand = XCHG->getPointerOperand();
      Value *Replacement =
          getTranslatedGlobalMemoryPtrOfLDS(LoadMallocPtr, XCHGPtrOperand);
      AtomicCmpXchgInst *NewXCHG = IRB.CreateAtomicCmpXchg(
          Replacement, XCHG->getCompareOperand(), XCHG->getNewValOperand(),
          XCHG->getAlign(), XCHG->getSuccessOrdering(),
          XCHG->getFailureOrdering(), XCHG->getSyncScopeID());
      NewXCHG->setVolatile(XCHG->isVolatile());
      AsanInfo.Instructions.insert(NewXCHG);
      XCHG->replaceAllUsesWith(NewXCHG);
      XCHG->eraseFromParent();
    } else if (AddrSpaceCastInst *ASC = dyn_cast<AddrSpaceCastInst>(Inst)) {
      Value *AIOperand = ASC->getPointerOperand();
      Value *Replacement =
          getTranslatedGlobalMemoryPtrOfLDS(LoadMallocPtr, AIOperand);
      Value *NewAI = IRB.CreateAddrSpaceCast(Replacement, ASC->getType());
      // Note: No need to add the instruction to AsanInfo instructions to be
      // instrumented list. FLAT_ADDRESS ptr would have been already
      // instrumented by asan pass prior to this pass.
      ASC->replaceAllUsesWith(NewAI);
      ASC->eraseFromParent();
    } else
      report_fatal_error("Unimplemented LDS lowering instruction");
  }
}

void AMDGPUSwLowerLDS::poisonRedzones(Function *Func, Value *MallocPtr) {
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  Type *Int64Ty = IRB.getInt64Ty();
  Type *VoidTy = IRB.getVoidTy();
  FunctionCallee AsanPoisonRegion = M.getOrInsertFunction(
      "__asan_poison_region",
      FunctionType::get(VoidTy, {Int64Ty, Int64Ty}, false));

  auto RedzonesVec = LDSParams.RedzoneOffsetAndSizeVector;
  size_t VecSize = RedzonesVec.size();
  for (unsigned i = 0; i < VecSize; i++) {
    auto &RedzonePair = RedzonesVec[i];
    uint64_t RedzoneOffset = RedzonePair.first;
    uint64_t RedzoneSize = RedzonePair.second;
    Value *RedzoneAddrOffset = IRB.CreateInBoundsGEP(
        IRB.getInt8Ty(), MallocPtr, {IRB.getInt64(RedzoneOffset)});
    Value *RedzoneAddress = IRB.CreatePtrToInt(RedzoneAddrOffset, Int64Ty);
    IRB.CreateCall(AsanPoisonRegion,
                   {RedzoneAddress, IRB.getInt64(RedzoneSize)});
  }
}

void AMDGPUSwLowerLDS::lowerKernelLDSAccesses(Function *Func,
                                              DomTreeUpdater &DTU) {
  LLVM_DEBUG(dbgs() << "Sw Lowering Kernel LDS for : " << Func->getName());
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
  auto &Ctx = M.getContext();
  auto *PrevEntryBlock = &Func->getEntryBlock();
  SetVector<Instruction *> LDSInstructions;
  getLDSMemoryInstructions(Func, LDSInstructions);

  // Create malloc block.
  auto *MallocBlock = BasicBlock::Create(Ctx, "Malloc", Func, PrevEntryBlock);

  // Create WIdBlock block which has instructions related to selection of
  // {0,0,0} indiex work item in the work group.
  auto *WIdBlock = BasicBlock::Create(Ctx, "WId", Func, MallocBlock);
  IRB.SetInsertPoint(WIdBlock, WIdBlock->begin());
  DebugLoc FirstDL =
      getOrCreateDebugLoc(&*PrevEntryBlock->begin(), Func->getSubprogram());
  IRB.SetCurrentDebugLocation(FirstDL);
  Value *WIdx = IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x, {}, {});
  Value *WIdy = IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_y, {}, {});
  Value *WIdz = IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_z, {}, {});
  Value *XYOr = IRB.CreateOr(WIdx, WIdy);
  Value *XYZOr = IRB.CreateOr(XYOr, WIdz);
  Value *WIdzCond = IRB.CreateICmpEQ(XYZOr, IRB.getInt32(0));

  // All work items will branch to PrevEntryBlock except {0,0,0} index
  // work item which will branch to malloc block.
  IRB.CreateCondBr(WIdzCond, MallocBlock, PrevEntryBlock);

  // Malloc block
  IRB.SetInsertPoint(MallocBlock, MallocBlock->begin());

  // If Dynamic LDS globals are accessed by the kernel,
  // Get the size of dyn lds from hidden dyn_lds_size kernel arg.
  // Update the corresponding metadata global entries for this dyn lds global.
  GlobalVariable *SwLDS = LDSParams.SwLDS;
  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDS && SwLDSMetadata);
  StructType *MetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  uint32_t MallocSize = 0;
  Value *CurrMallocSize;
  Type *Int32Ty = IRB.getInt32Ty();
  Type *Int64Ty = IRB.getInt64Ty();

  SetVector<GlobalVariable *> UniqueLDSGlobals;
  auto GetUniqueLDSGlobals = [&](SetVector<GlobalVariable *> &LDSGlobals) {
    for (auto &GV : LDSGlobals) {
      if (is_contained(UniqueLDSGlobals, GV))
        continue;
      UniqueLDSGlobals.insert(GV);
    }
  };

  GetUniqueLDSGlobals(LDSParams.DirectAccess.StaticLDSGlobals);
  GetUniqueLDSGlobals(LDSParams.IndirectAccess.StaticLDSGlobals);
  unsigned NumStaticLDS = 1 + UniqueLDSGlobals.size();
  UniqueLDSGlobals.clear();

  if (NumStaticLDS) {
    auto *GEPForEndStaticLDSOffset =
        IRB.CreateInBoundsGEP(MetadataStructType, SwLDSMetadata,
                              {ConstantInt::get(Int32Ty, 0),
                               ConstantInt::get(Int32Ty, NumStaticLDS - 1),
                               ConstantInt::get(Int32Ty, 0)});

    auto *GEPForEndStaticLDSSize =
        IRB.CreateInBoundsGEP(MetadataStructType, SwLDSMetadata,
                              {ConstantInt::get(Int32Ty, 0),
                               ConstantInt::get(Int32Ty, NumStaticLDS - 1),
                               ConstantInt::get(Int32Ty, 2)});

    Value *EndStaticLDSOffset =
        IRB.CreateLoad(Int32Ty, GEPForEndStaticLDSOffset);
    Value *EndStaticLDSSize = IRB.CreateLoad(Int32Ty, GEPForEndStaticLDSSize);
    CurrMallocSize = IRB.CreateAdd(EndStaticLDSOffset, EndStaticLDSSize);
  } else
    CurrMallocSize = IRB.getInt32(MallocSize);

  if (LDSParams.SwDynLDS) {
    if (!(AMDGPU::getAMDHSACodeObjectVersion(M) >= AMDGPU::AMDHSA_COV5))
      report_fatal_error(
          "Dynamic LDS size query is only supported for CO V5 and later.");
    // Get size from hidden dyn_lds_size argument of kernel
    Value *ImplicitArg =
        IRB.CreateIntrinsic(Intrinsic::amdgcn_implicitarg_ptr, {}, {});
    Value *HiddenDynLDSSize = IRB.CreateInBoundsGEP(
        ImplicitArg->getType(), ImplicitArg,
        {ConstantInt::get(Int64Ty, COV5_HIDDEN_DYN_LDS_SIZE_ARG)});
    UniqueLDSGlobals.clear();
    GetUniqueLDSGlobals(LDSParams.DirectAccess.DynamicLDSGlobals);
    GetUniqueLDSGlobals(LDSParams.IndirectAccess.DynamicLDSGlobals);
    updateMallocSizeForDynamicLDS(Func, &CurrMallocSize, HiddenDynLDSSize,
                                  UniqueLDSGlobals);
  }

  CurrMallocSize = IRB.CreateZExt(CurrMallocSize, Int64Ty);

  // Create a call to malloc function which does device global memory allocation
  // with size equals to all LDS global accesses size in this kernel.
  Value *ReturnAddress =
      IRB.CreateIntrinsic(Intrinsic::returnaddress, {}, {IRB.getInt32(0)});
  FunctionCallee MallocFunc = M.getOrInsertFunction(
      StringRef("__asan_malloc_impl"),
      FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false));
  Value *RAPtrToInt = IRB.CreatePtrToInt(ReturnAddress, Int64Ty);
  Value *MallocCall = IRB.CreateCall(MallocFunc, {CurrMallocSize, RAPtrToInt});

  Value *MallocPtr =
      IRB.CreateIntToPtr(MallocCall, IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS));

  // Create store of malloc to new global
  IRB.CreateStore(MallocPtr, SwLDS);

  // Create calls to __asan_poison_region to poison redzones.
  poisonRedzones(Func, MallocPtr);

  // Create branch to PrevEntryBlock
  IRB.CreateBr(PrevEntryBlock);

  // Create wave-group barrier at the starting of Previous entry block
  Type *Int1Ty = IRB.getInt1Ty();
  IRB.SetInsertPoint(PrevEntryBlock, PrevEntryBlock->begin());
  auto *XYZCondPhi = IRB.CreatePHI(Int1Ty, 2, "xyzCond");
  XYZCondPhi->addIncoming(IRB.getInt1(0), WIdBlock);
  XYZCondPhi->addIncoming(IRB.getInt1(1), MallocBlock);

  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  // Load malloc pointer from Sw LDS.
  Value *LoadMallocPtr =
      IRB.CreateLoad(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), SwLDS);

  // Replace All uses of LDS globals with new LDS pointers.
  replaceKernelLDSAccesses(Func);

  // Replace Memory Operations on LDS with corresponding
  // global memory pointers.
  translateLDSMemoryOperationsToGlobalMemory(Func, LoadMallocPtr,
                                             LDSInstructions);

  auto *CondFreeBlock = BasicBlock::Create(Ctx, "CondFree", Func);
  auto *FreeBlock = BasicBlock::Create(Ctx, "Free", Func);
  auto *EndBlock = BasicBlock::Create(Ctx, "End", Func);
  for (BasicBlock &BB : *Func) {
    if (!BB.empty()) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&BB.back())) {
        RI->eraseFromParent();
        IRB.SetInsertPoint(&BB, BB.end());
        IRB.CreateBr(CondFreeBlock);
      }
    }
  }

  // Cond Free Block
  IRB.SetInsertPoint(CondFreeBlock, CondFreeBlock->begin());
  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});
  IRB.CreateCondBr(XYZCondPhi, FreeBlock, EndBlock);

  // Free Block
  IRB.SetInsertPoint(FreeBlock, FreeBlock->begin());

  // Free the previously allocate device global memory.
  FunctionCallee AsanFreeFunc = M.getOrInsertFunction(
      StringRef("__asan_free_impl"),
      FunctionType::get(IRB.getVoidTy(), {Int64Ty, Int64Ty}, false));
  Value *ReturnAddr =
      IRB.CreateIntrinsic(Intrinsic::returnaddress, {}, IRB.getInt32(0));
  Value *RAPToInt = IRB.CreatePtrToInt(ReturnAddr, Int64Ty);
  Value *MallocPtrToInt = IRB.CreatePtrToInt(LoadMallocPtr, Int64Ty);
  IRB.CreateCall(AsanFreeFunc, {MallocPtrToInt, RAPToInt});

  IRB.CreateBr(EndBlock);

  // End Block
  IRB.SetInsertPoint(EndBlock, EndBlock->begin());
  IRB.CreateRetVoid();
  // Update the DomTree with corresponding links to basic blocks.
  DTU.applyUpdates({{DominatorTree::Insert, WIdBlock, MallocBlock},
                    {DominatorTree::Insert, MallocBlock, PrevEntryBlock},
                    {DominatorTree::Insert, CondFreeBlock, FreeBlock},
                    {DominatorTree::Insert, FreeBlock, EndBlock}});
}

Constant *AMDGPUSwLowerLDS::getAddressesOfVariablesInKernel(
    Function *Func, SetVector<GlobalVariable *> &Variables) {
  Type *Int32Ty = IRB.getInt32Ty();
  auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];

  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDSMetadata);
  auto *SwLDSMetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  ArrayType *KernelOffsetsType =
      ArrayType::get(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), Variables.size());

  SmallVector<Constant *> Elements;
  for (auto *GV : Variables) {
    if (!LDSParams.LDSToReplacementIndicesMap.contains(GV)) {
      Elements.push_back(
          PoisonValue::get(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS)));
      continue;
    }
    auto &Indices = LDSParams.LDSToReplacementIndicesMap[GV];
    Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Indices[0]),
                          ConstantInt::get(Int32Ty, Indices[1]),
                          ConstantInt::get(Int32Ty, Indices[2])};
    Constant *GEP = ConstantExpr::getGetElementPtr(SwLDSMetadataStructType,
                                                   SwLDSMetadata, GEPIdx, true);
    Elements.push_back(GEP);
  }
  return ConstantArray::get(KernelOffsetsType, Elements);
}

void AMDGPUSwLowerLDS::buildNonKernelLDSBaseTable(
    NonKernelLDSParameters &NKLDSParams) {
  // Base table will have single row, with elements of the row
  // placed as per kernel ID. Each element in the row corresponds
  // to addresss of "SW LDS" global of the kernel.
  auto &Kernels = NKLDSParams.OrderedKernels;
  if (Kernels.empty())
    return;
  Type *Int32Ty = IRB.getInt32Ty();
  const size_t NumberKernels = Kernels.size();
  ArrayType *AllKernelsOffsetsType =
      ArrayType::get(IRB.getPtrTy(AMDGPUAS::LOCAL_ADDRESS), NumberKernels);
  std::vector<Constant *> OverallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
    GlobalVariable *SwLDS = LDSParams.SwLDS;
    assert(SwLDS);
    Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, 0)};
    Constant *GEP =
        ConstantExpr::getGetElementPtr(SwLDS->getType(), SwLDS, GEPIdx, true);
    OverallConstantExprElts[i] = GEP;
  }
  Constant *init =
      ConstantArray::get(AllKernelsOffsetsType, OverallConstantExprElts);
  NKLDSParams.LDSBaseTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, init,
      "llvm.amdgcn.sw.lds.base.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::GLOBAL_ADDRESS);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  NKLDSParams.LDSBaseTable->setSanitizerMetadata(MD);
}

void AMDGPUSwLowerLDS::buildNonKernelLDSOffsetTable(
    NonKernelLDSParameters &NKLDSParams) {
  // Offset table will have multiple rows and columns.
  // Rows are assumed to be from 0 to (n-1). n is total number
  // of kernels accessing the LDS through non-kernels.
  // Each row will have m elements. m is the total number of
  // unique LDS globals accessed by non-kernels.
  // Each element in the row correspond to the address of
  // the replacement of LDS global done by that particular kernel.
  auto &Variables = NKLDSParams.OrdereLDSGlobals;
  auto &Kernels = NKLDSParams.OrderedKernels;
  if (Variables.empty() || Kernels.empty())
    return;
  const size_t NumberVariables = Variables.size();
  const size_t NumberKernels = Kernels.size();

  ArrayType *KernelOffsetsType =
      ArrayType::get(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), NumberVariables);

  ArrayType *AllKernelsOffsetsType =
      ArrayType::get(KernelOffsetsType, NumberKernels);
  std::vector<Constant *> overallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    overallConstantExprElts[i] =
        getAddressesOfVariablesInKernel(Func, Variables);
  }
  Constant *Init =
      ConstantArray::get(AllKernelsOffsetsType, overallConstantExprElts);
  NKLDSParams.LDSOffsetTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, Init,
      "llvm.amdgcn.sw.lds.offset.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::GLOBAL_ADDRESS);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  NKLDSParams.LDSOffsetTable->setSanitizerMetadata(MD);
}

void AMDGPUSwLowerLDS::lowerNonKernelLDSAccesses(
    Function *Func, SetVector<GlobalVariable *> &LDSGlobals,
    NonKernelLDSParameters &NKLDSParams) {
  // Replace LDS access in non-kernel with replacement queried from
  // Base table and offset from offset table.
  LLVM_DEBUG(dbgs() << "Sw LDS lowering, lower non-kernel access for : "
                    << Func->getName());
  auto InsertAt = Func->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
  IRB.SetInsertPoint(InsertAt);

  // Get LDS memory instructions.
  SetVector<Instruction *> LDSInstructions;
  getLDSMemoryInstructions(Func, LDSInstructions);

  auto *KernelId = IRB.CreateIntrinsic(Intrinsic::amdgcn_lds_kernel_id, {}, {});
  GlobalVariable *LDSBaseTable = NKLDSParams.LDSBaseTable;
  GlobalVariable *LDSOffsetTable = NKLDSParams.LDSOffsetTable;
  auto &OrdereLDSGlobals = NKLDSParams.OrdereLDSGlobals;
  Value *BaseGEP = IRB.CreateInBoundsGEP(
      LDSBaseTable->getValueType(), LDSBaseTable, {IRB.getInt32(0), KernelId});
  Value *BaseLoad =
      IRB.CreateLoad(IRB.getPtrTy(AMDGPUAS::LOCAL_ADDRESS), BaseGEP);
  Value *LoadMallocPtr =
      IRB.CreateLoad(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), BaseLoad);

  for (GlobalVariable *GV : LDSGlobals) {
    const auto *GVIt =
        std::find(OrdereLDSGlobals.begin(), OrdereLDSGlobals.end(), GV);
    assert(GVIt != OrdereLDSGlobals.end());
    uint32_t GVOffset = std::distance(OrdereLDSGlobals.begin(), GVIt);

    Value *OffsetGEP = IRB.CreateInBoundsGEP(
        LDSOffsetTable->getValueType(), LDSOffsetTable,
        {IRB.getInt32(0), KernelId, IRB.getInt32(GVOffset)});
    Value *OffsetLoad =
        IRB.CreateLoad(IRB.getPtrTy(AMDGPUAS::GLOBAL_ADDRESS), OffsetGEP);
    Value *Offset = IRB.CreateLoad(IRB.getInt32Ty(), OffsetLoad);
    Value *BasePlusOffset =
        IRB.CreateInBoundsGEP(IRB.getInt8Ty(), BaseLoad, {Offset});
    LLVM_DEBUG(dbgs() << "Sw LDS Lowering, Replace non-kernel LDS for "
                      << GV->getName());
    replacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
  }
  translateLDSMemoryOperationsToGlobalMemory(Func, LoadMallocPtr,
                                             LDSInstructions);
}

static void reorderStaticDynamicIndirectLDSSet(KernelLDSParameters &LDSParams) {
  // Sort Static, dynamic LDS globals which are either
  // direct or indirect access on basis of name.
  auto &DirectAccess = LDSParams.DirectAccess;
  auto &IndirectAccess = LDSParams.IndirectAccess;
  LDSParams.DirectAccess.StaticLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(DirectAccess.StaticLDSGlobals.begin(),
                                    DirectAccess.StaticLDSGlobals.end()));
  LDSParams.DirectAccess.DynamicLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(DirectAccess.DynamicLDSGlobals.begin(),
                                    DirectAccess.DynamicLDSGlobals.end()));
  LDSParams.IndirectAccess.StaticLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(IndirectAccess.StaticLDSGlobals.begin(),
                                    IndirectAccess.StaticLDSGlobals.end()));
  LDSParams.IndirectAccess.DynamicLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(IndirectAccess.DynamicLDSGlobals.begin(),
                                    IndirectAccess.DynamicLDSGlobals.end()));
}

void AMDGPUSwLowerLDS::initAsanInfo() {
  // Get Shadow mapping scale and offset.
  unsigned LongSize =
      M.getDataLayout().getPointerSizeInBits(AMDGPUAS::GLOBAL_ADDRESS);
  uint64_t Offset;
  int Scale;
  bool OrShadowOffset;
  llvm::getAddressSanitizerParams(Triple(AMDGPUTM.getTargetTriple()), LongSize,
                                  false, &Offset, &Scale, &OrShadowOffset);
  AsanInfo.Scale = Scale;
  AsanInfo.Offset = Offset;
}

bool AMDGPUSwLowerLDS::run() {
  bool Changed = false;

  CallGraph CG = CallGraph(M);

  Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

  // Get all the direct and indirect access of LDS for all the kernels.
  LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

  // Utility to group LDS access into direct, indirect, static and dynamic.
  auto PopulateKernelStaticDynamicLDS = [&](FunctionVariableMap &LDSAccesses,
                                            bool DirectAccess) {
    for (auto &K : LDSAccesses) {
      Function *F = K.first;
      if (!F || K.second.empty())
        continue;

      assert(isKernelLDS(F));
      if (!F->hasFnAttribute(Attribute::SanitizeAddress))
        continue;

      // Only inserts if key isn't already in the map.
      FuncLDSAccessInfo.KernelToLDSParametersMap.insert(
          {F, KernelLDSParameters()});

      auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[F];
      if (!DirectAccess)
        FuncLDSAccessInfo.KernelsWithIndirectLDSAccess.insert(F);
      for (GlobalVariable *GV : K.second) {
        if (!DirectAccess) {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams.IndirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams.IndirectAccess.StaticLDSGlobals.insert(GV);
          FuncLDSAccessInfo.AllNonKernelLDSAccess.insert(GV);
        } else {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams.DirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams.DirectAccess.StaticLDSGlobals.insert(GV);
        }
      }
    }
  };

  PopulateKernelStaticDynamicLDS(LDSUsesInfo.direct_access, true);
  PopulateKernelStaticDynamicLDS(LDSUsesInfo.indirect_access, false);

  // Get address sanitizer scale.
  initAsanInfo();

  for (auto &K : FuncLDSAccessInfo.KernelToLDSParametersMap) {
    Function *Func = K.first;
    auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
    if (LDSParams.DirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.DirectAccess.DynamicLDSGlobals.empty() &&
        LDSParams.IndirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.IndirectAccess.DynamicLDSGlobals.empty()) {
      Changed = false;
    } else {
      removeFnAttrFromReachable(
          CG, Func,
          {"amdgpu-no-workitem-id-x", "amdgpu-no-workitem-id-y",
           "amdgpu-no-workitem-id-z", "amdgpu-no-heap-ptr"});
      if (!LDSParams.IndirectAccess.StaticLDSGlobals.empty() ||
          !LDSParams.IndirectAccess.DynamicLDSGlobals.empty())
        removeFnAttrFromReachable(CG, Func, {"amdgpu-no-lds-kernel-id"});
      reorderStaticDynamicIndirectLDSSet(LDSParams);
      buildSwLDSGlobal(Func);
      buildSwDynLDSGlobal(Func);
      populateSwMetadataGlobal(Func);
      populateSwLDSAttributeAndMetadata(Func);
      populateLDSToReplacementIndicesMap(Func);
      DomTreeUpdater DTU(DTCallback(*Func),
                         DomTreeUpdater::UpdateStrategy::Lazy);
      lowerKernelLDSAccesses(Func, DTU);
      Changed = true;
    }
  }

  // Get the Uses of LDS from non-kernels.
  getUsesOfLDSByNonKernels();

  // Get non-kernels with LDS ptr as argument and called by kernels.
  getNonKernelsWithLDSArguments(CG);

  if (!FuncLDSAccessInfo.NonKernelToLDSAccessMap.empty() ||
      !FuncLDSAccessInfo.NonKernelsWithLDSArgument.empty()) {
    NonKernelLDSParameters NKLDSParams;
    NKLDSParams.OrderedKernels = getOrderedIndirectLDSAccessingKernels(
        FuncLDSAccessInfo.KernelsWithIndirectLDSAccess);
    NKLDSParams.OrdereLDSGlobals = getOrderedNonKernelAllLDSGlobals(
        FuncLDSAccessInfo.AllNonKernelLDSAccess);
    buildNonKernelLDSBaseTable(NKLDSParams);
    buildNonKernelLDSOffsetTable(NKLDSParams);
    for (auto &K : FuncLDSAccessInfo.NonKernelToLDSAccessMap) {
      Function *Func = K.first;
      DenseSet<GlobalVariable *> &LDSGlobals = K.second;
      SetVector<GlobalVariable *> OrderedLDSGlobals = sortByName(
          std::vector<GlobalVariable *>(LDSGlobals.begin(), LDSGlobals.end()));
      lowerNonKernelLDSAccesses(Func, OrderedLDSGlobals, NKLDSParams);
    }
    for (Function *Func : FuncLDSAccessInfo.NonKernelsWithLDSArgument) {
      auto &K = FuncLDSAccessInfo.NonKernelToLDSAccessMap;
      if (K.find(Func) != K.end())
        continue;
      SetVector<llvm::GlobalVariable *> Vec;
      lowerNonKernelLDSAccesses(Func, Vec, NKLDSParams);
    }
    Changed = true;
  }

  if (!Changed)
    return Changed;

  for (auto &GV : make_early_inc_range(M.globals())) {
    if (AMDGPU::isLDSVariableToLower(GV)) {
      // probably want to remove from used lists
      GV.removeDeadConstantUsers();
      if (GV.use_empty())
        GV.eraseFromParent();
    }
  }

  if (AsanInstrumentLDS) {
    SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
    for (Instruction *Inst : AsanInfo.Instructions) {
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInterestingMemoryOperands(M, Inst, InterestingOperands);
      for (auto &Operand : InterestingOperands) {
        OperandsToInstrument.push_back(Operand);
      }
    }
    for (auto &Operand : OperandsToInstrument) {
      Value *Addr = Operand.getPtr();
      instrumentAddress(M, IRB, Operand.getInsn(), Operand.getInsn(), Addr,
                        Operand.Alignment.valueOrOne(), Operand.TypeStoreSize,
                        Operand.IsWrite, nullptr, false, false, AsanInfo.Scale,
                        AsanInfo.Offset);
      Changed = true;
    }
  }

  return Changed;
}

class AMDGPUSwLowerLDSLegacy : public ModulePass {
public:
  const AMDGPUTargetMachine *AMDGPUTM;
  static char ID;
  AMDGPUSwLowerLDSLegacy(const AMDGPUTargetMachine *TM)
      : ModulePass(ID), AMDGPUTM(TM) {
    initializeAMDGPUSwLowerLDSLegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
} // namespace

char AMDGPUSwLowerLDSLegacy::ID = 0;
char &llvm::AMDGPUSwLowerLDSLegacyPassID = AMDGPUSwLowerLDSLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUSwLowerLDSLegacy, "amdgpu-sw-lower-lds",
                      "AMDGPU Software lowering of LDS", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUSwLowerLDSLegacy, "amdgpu-sw-lower-lds",
                    "AMDGPU Software lowering of LDS", false, false)

bool AMDGPUSwLowerLDSLegacy::runOnModule(Module &M) {
  // AddressSanitizer pass adds "nosanitize_address" module flag if it has
  // instrumented the IR. Return early if the flag is not present.
  if (!M.getModuleFlag("nosanitize_address"))
    return false;
  DominatorTreeWrapperPass *const DTW =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto DTCallback = [&DTW](Function &F) -> DominatorTree * {
    return DTW ? &DTW->getDomTree() : nullptr;
  };
  if (!AMDGPUTM) {
    auto &TPC = getAnalysis<TargetPassConfig>();
    AMDGPUTM = &TPC.getTM<AMDGPUTargetMachine>();
  }
  AMDGPUSwLowerLDS SwLowerLDSImpl(M, *AMDGPUTM, DTCallback);
  bool IsChanged = SwLowerLDSImpl.run();
  return IsChanged;
}

ModulePass *
llvm::createAMDGPUSwLowerLDSLegacyPass(const AMDGPUTargetMachine *TM) {
  return new AMDGPUSwLowerLDSLegacy(TM);
}

PreservedAnalyses AMDGPUSwLowerLDSPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  // AddressSanitizer pass adds "nosanitize_address" module flag if it has
  // instrumented the IR. Return early if the flag is not present.
  if (!M.getModuleFlag("nosanitize_address"))
    return PreservedAnalyses::all();
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto DTCallback = [&FAM](Function &F) -> DominatorTree * {
    return &FAM.getResult<DominatorTreeAnalysis>(F);
  };
  AMDGPUSwLowerLDS SwLowerLDSImpl(M, TM, DTCallback);
  bool IsChanged = SwLowerLDSImpl.run();
  if (!IsChanged)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
