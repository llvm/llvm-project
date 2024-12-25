//===----- TypeSanitizer.cpp - type-based-aliasing-violation detector -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TypeSanitizer, a type-based-aliasing-violation
// detector.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/TypeSanitizer.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cctype>

using namespace llvm;

#define DEBUG_TYPE "tysan"

static const char *const kTysanModuleCtorName = "tysan.module_ctor";
static const char *const kTysanInitName = "__tysan_init";
static const char *const kTysanCheckName = "__tysan_check";
static const char *const kTysanGVNamePrefix = "__tysan_v1_";

static const char *const kTysanShadowMemoryAddress =
    "__tysan_shadow_memory_address";
static const char *const kTysanAppMemMask = "__tysan_app_memory_mask";

static cl::opt<bool>
    ClWritesAlwaysSetType("tysan-writes-always-set-type",
                          cl::desc("Writes always set the type"), cl::Hidden,
                          cl::init(false));

STATISTIC(NumInstrumentedAccesses, "Number of instrumented accesses");

namespace {

/// TypeSanitizer: instrument the code in module to find type-based aliasing
/// violations.
struct TypeSanitizer {
  TypeSanitizer(Module &M);
  bool run(Function &F, const TargetLibraryInfo &TLI);
  void instrumentGlobals(Module &M);

private:
  typedef SmallDenseMap<const MDNode *, GlobalVariable *, 8>
      TypeDescriptorsMapTy;
  typedef SmallDenseMap<const MDNode *, std::string, 8> TypeNameMapTy;

  void initializeCallbacks(Module &M);

  Instruction *getShadowBase(Function &F);
  Instruction *getAppMemMask(Function &F);

  bool instrumentWithShadowUpdate(IRBuilder<> &IRB, const MDNode *TBAAMD,
                                  Value *Ptr, uint64_t AccessSize, bool IsRead,
                                  bool IsWrite, Value *ShadowBase,
                                  Value *AppMemMask, bool ForceSetType,
                                  bool SanitizeFunction,
                                  TypeDescriptorsMapTy &TypeDescriptors,
                                  const DataLayout &DL);

  /// Memory-related intrinsics/instructions reset the type of the destination
  /// memory (including allocas and byval arguments).
  bool instrumentMemInst(Value *I, Instruction *ShadowBase,
                         Instruction *AppMemMask, const DataLayout &DL);

  std::string getAnonymousStructIdentifier(const MDNode *MD,
                                           TypeNameMapTy &TypeNames);
  bool generateTypeDescriptor(const MDNode *MD,
                              TypeDescriptorsMapTy &TypeDescriptors,
                              TypeNameMapTy &TypeNames, Module &M);
  bool generateBaseTypeDescriptor(const MDNode *MD,
                                  TypeDescriptorsMapTy &TypeDescriptors,
                                  TypeNameMapTy &TypeNames, Module &M);

  const Triple TargetTriple;
  Regex AnonNameRegex;
  Type *IntptrTy;
  uint64_t PtrShift;
  IntegerType *OrdTy;

  /// Callbacks to run-time library are computed in initializeCallbacks.
  FunctionCallee TysanCheck;
  FunctionCallee TysanCtorFunction;

  /// Callback to set types for gloabls.
  Function *TysanGlobalsSetTypeFunction;
};
} // namespace

TypeSanitizer::TypeSanitizer(Module &M)
    : TargetTriple(Triple(M.getTargetTriple())),
      AnonNameRegex("^_ZTS.*N[1-9][0-9]*_GLOBAL__N") {
  const DataLayout &DL = M.getDataLayout();
  IntptrTy = DL.getIntPtrType(M.getContext());
  PtrShift = countr_zero(IntptrTy->getPrimitiveSizeInBits() / 8);

  TysanGlobalsSetTypeFunction = M.getFunction("__tysan_set_globals_types");
  initializeCallbacks(M);
}

void TypeSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  OrdTy = IRB.getInt32Ty();

  AttributeList Attr;
  Attr = Attr.addFnAttribute(M.getContext(), Attribute::NoUnwind);
  // Initialize the callbacks.
  TysanCheck =
      M.getOrInsertFunction(kTysanCheckName, Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), // Pointer to data to be read.
                            OrdTy,          // Size of the data in bytes.
                            IRB.getPtrTy(), // Pointer to type descriptor.
                            OrdTy           // Flags.
      );

  TysanCtorFunction =
      M.getOrInsertFunction(kTysanModuleCtorName, Attr, IRB.getVoidTy());
}

void TypeSanitizer::instrumentGlobals(Module &M) {
  TysanGlobalsSetTypeFunction = nullptr;

  NamedMDNode *Globals = M.getNamedMetadata("llvm.tysan.globals");
  if (!Globals)
    return;

  TysanGlobalsSetTypeFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(M.getContext()), false),
      GlobalValue::InternalLinkage, "__tysan_set_globals_types", &M);
  BasicBlock *BB =
      BasicBlock::Create(M.getContext(), "", TysanGlobalsSetTypeFunction);
  ReturnInst::Create(M.getContext(), BB);

  const DataLayout &DL = M.getDataLayout();
  Value *ShadowBase = getShadowBase(*TysanGlobalsSetTypeFunction);
  Value *AppMemMask = getAppMemMask(*TysanGlobalsSetTypeFunction);
  TypeDescriptorsMapTy TypeDescriptors;
  TypeNameMapTy TypeNames;

  for (const auto &GMD : Globals->operands()) {
    auto *GV = mdconst::dyn_extract_or_null<GlobalVariable>(GMD->getOperand(0));
    if (!GV)
      continue;
    const MDNode *TBAAMD = cast<MDNode>(GMD->getOperand(1));
    if (!generateBaseTypeDescriptor(TBAAMD, TypeDescriptors, TypeNames, M))
      continue;

    IRBuilder<> IRB(
        TysanGlobalsSetTypeFunction->getEntryBlock().getTerminator());
    Type *AccessTy = GV->getValueType();
    assert(AccessTy->isSized());
    uint64_t AccessSize = DL.getTypeStoreSize(AccessTy);
    instrumentWithShadowUpdate(IRB, TBAAMD, GV, AccessSize, false, false,
                               ShadowBase, AppMemMask, true, false,
                               TypeDescriptors, DL);
  }

  if (TysanGlobalsSetTypeFunction) {
    IRBuilder<> IRB(cast<Function>(TysanCtorFunction.getCallee())
                        ->getEntryBlock()
                        .getTerminator());
    IRB.CreateCall(TysanGlobalsSetTypeFunction, {});
  }
}

static const char LUT[] = "0123456789abcdef";

static std::string encodeName(StringRef Name) {
  size_t Length = Name.size();
  std::string Output = kTysanGVNamePrefix;
  Output.reserve(Output.size() + 3 * Length);
  for (size_t i = 0; i < Length; ++i) {
    const unsigned char c = Name[i];
    if (isalnum(c)) {
      Output.push_back(c);
      continue;
    }

    if (c == '_') {
      Output.append("__");
      continue;
    }

    Output.push_back('_');
    Output.push_back(LUT[c >> 4]);
    Output.push_back(LUT[c & 15]);
  }

  return Output;
}

std::string
TypeSanitizer::getAnonymousStructIdentifier(const MDNode *MD,
                                            TypeNameMapTy &TypeNames) {
  MD5 Hash;

  for (int i = 1, e = MD->getNumOperands(); i < e; i += 2) {
    const MDNode *MemberNode = dyn_cast<MDNode>(MD->getOperand(i));
    if (!MemberNode)
      return "";

    auto TNI = TypeNames.find(MemberNode);
    std::string MemberName;
    if (TNI != TypeNames.end()) {
      MemberName = TNI->second;
    } else {
      if (MemberNode->getNumOperands() < 1)
        return "";
      MDString *MemberNameNode = dyn_cast<MDString>(MemberNode->getOperand(0));
      if (!MemberNameNode)
        return "";
      MemberName = MemberNameNode->getString().str();
      if (MemberName.empty())
        MemberName = getAnonymousStructIdentifier(MemberNode, TypeNames);
      if (MemberName.empty())
        return "";
      TypeNames[MemberNode] = MemberName;
    }

    Hash.update(MemberName);
    Hash.update("\0");

    uint64_t Offset =
        mdconst::extract<ConstantInt>(MD->getOperand(i + 1))->getZExtValue();
    Hash.update(utostr(Offset));
    Hash.update("\0");
  }

  MD5::MD5Result HashResult;
  Hash.final(HashResult);
  return "__anonymous_" + std::string(HashResult.digest().str());
}

bool TypeSanitizer::generateBaseTypeDescriptor(
    const MDNode *MD, TypeDescriptorsMapTy &TypeDescriptors,
    TypeNameMapTy &TypeNames, Module &M) {
  if (MD->getNumOperands() < 1)
    return false;

  MDString *NameNode = dyn_cast<MDString>(MD->getOperand(0));
  if (!NameNode)
    return false;

  std::string Name = NameNode->getString().str();
  if (Name.empty())
    Name = getAnonymousStructIdentifier(MD, TypeNames);
  if (Name.empty())
    return false;
  TypeNames[MD] = Name;
  std::string EncodedName = encodeName(Name);

  GlobalVariable *GV =
      dyn_cast_or_null<GlobalVariable>(M.getNamedValue(EncodedName));
  if (GV) {
    TypeDescriptors[MD] = GV;
    return true;
  }

  SmallVector<std::pair<Constant *, uint64_t>> Members;
  for (int i = 1, e = MD->getNumOperands(); i < e; i += 2) {
    const MDNode *MemberNode = dyn_cast<MDNode>(MD->getOperand(i));
    if (!MemberNode)
      return false;

    Constant *Member;
    auto TDI = TypeDescriptors.find(MemberNode);
    if (TDI != TypeDescriptors.end()) {
      Member = TDI->second;
    } else {
      if (!generateBaseTypeDescriptor(MemberNode, TypeDescriptors, TypeNames,
                                      M))
        return false;

      Member = TypeDescriptors[MemberNode];
    }

    uint64_t Offset =
        mdconst::extract<ConstantInt>(MD->getOperand(i + 1))->getZExtValue();

    Members.push_back(std::make_pair(Member, Offset));
  }

  // The descriptor for a scalar is:
  //   [2, member count, [type pointer, offset]..., name]

  LLVMContext &C = MD->getContext();
  Constant *NameData = ConstantDataArray::getString(C, NameNode->getString());
  SmallVector<Type *> TDSubTys;
  SmallVector<Constant *> TDSubData;

  auto PushTDSub = [&](Constant *C) {
    TDSubTys.push_back(C->getType());
    TDSubData.push_back(C);
  };

  PushTDSub(ConstantInt::get(IntptrTy, 2));
  PushTDSub(ConstantInt::get(IntptrTy, Members.size()));

  // Types that are in an anonymous namespace are local to this module.
  // FIXME: This should really be marked by the frontend in the metadata
  // instead of having us guess this from the mangled name. Moreover, the regex
  // here can pick up (unlikely) names in the non-reserved namespace (because
  // it needs to search into the type to pick up cases where the type in the
  // anonymous namespace is a template parameter, etc.).
  bool ShouldBeComdat = !AnonNameRegex.match(NameNode->getString());
  for (auto &Member : Members) {
    PushTDSub(Member.first);
    PushTDSub(ConstantInt::get(IntptrTy, Member.second));
  }

  PushTDSub(NameData);

  StructType *TDTy = StructType::get(C, TDSubTys);
  Constant *TD = ConstantStruct::get(TDTy, TDSubData);

  GlobalVariable *TDGV =
      new GlobalVariable(TDTy, true,
                         !ShouldBeComdat ? GlobalValue::InternalLinkage
                                         : GlobalValue::LinkOnceODRLinkage,
                         TD, EncodedName);
  M.insertGlobalVariable(TDGV);

  if (ShouldBeComdat) {
    if (TargetTriple.isOSBinFormatELF()) {
      Comdat *TDComdat = M.getOrInsertComdat(EncodedName);
      TDGV->setComdat(TDComdat);
    }
    appendToUsed(M, TDGV);
  }

  TypeDescriptors[MD] = TDGV;
  return true;
}

bool TypeSanitizer::generateTypeDescriptor(
    const MDNode *MD, TypeDescriptorsMapTy &TypeDescriptors,
    TypeNameMapTy &TypeNames, Module &M) {
  // Here we need to generate a type descriptor corresponding to this TBAA
  // metadata node. Under the current scheme there are three kinds of TBAA
  // metadata nodes: scalar nodes, struct nodes, and struct tag nodes.

  if (MD->getNumOperands() < 3)
    return false;

  const MDNode *BaseNode = dyn_cast<MDNode>(MD->getOperand(0));
  if (!BaseNode)
    return false;

  // This is a struct tag (element-access) node.

  const MDNode *AccessNode = dyn_cast<MDNode>(MD->getOperand(1));
  if (!AccessNode)
    return false;

  Constant *Base;
  auto TDI = TypeDescriptors.find(BaseNode);
  if (TDI != TypeDescriptors.end()) {
    Base = TDI->second;
  } else {
    if (!generateBaseTypeDescriptor(BaseNode, TypeDescriptors, TypeNames, M))
      return false;

    Base = TypeDescriptors[BaseNode];
  }

  Constant *Access;
  TDI = TypeDescriptors.find(AccessNode);
  if (TDI != TypeDescriptors.end()) {
    Access = TDI->second;
  } else {
    if (!generateBaseTypeDescriptor(AccessNode, TypeDescriptors, TypeNames, M))
      return false;

    Access = TypeDescriptors[AccessNode];
  }

  uint64_t Offset =
      mdconst::extract<ConstantInt>(MD->getOperand(2))->getZExtValue();
  std::string EncodedName =
      std::string(Base->getName()) + "_o_" + utostr(Offset);

  GlobalVariable *GV =
      dyn_cast_or_null<GlobalVariable>(M.getNamedValue(EncodedName));
  if (GV) {
    TypeDescriptors[MD] = GV;
    return true;
  }

  // The descriptor for a scalar is:
  //   [1, base-type pointer, access-type pointer, offset]

  StructType *TDTy =
      StructType::get(IntptrTy, Base->getType(), Access->getType(), IntptrTy);
  Constant *TD =
      ConstantStruct::get(TDTy, ConstantInt::get(IntptrTy, 1), Base, Access,
                          ConstantInt::get(IntptrTy, Offset));

  bool ShouldBeComdat = cast<GlobalVariable>(Base)->getLinkage() ==
                        GlobalValue::LinkOnceODRLinkage;

  GlobalVariable *TDGV =
      new GlobalVariable(TDTy, true,
                         !ShouldBeComdat ? GlobalValue::InternalLinkage
                                         : GlobalValue::LinkOnceODRLinkage,
                         TD, EncodedName);
  M.insertGlobalVariable(TDGV);

  if (ShouldBeComdat) {
    if (TargetTriple.isOSBinFormatELF()) {
      Comdat *TDComdat = M.getOrInsertComdat(EncodedName);
      TDGV->setComdat(TDComdat);
    }
    appendToUsed(M, TDGV);
  }

  TypeDescriptors[MD] = TDGV;
  return true;
}

Instruction *TypeSanitizer::getShadowBase(Function &F) {
  IRBuilder<> IRB(&F.front().front());
  Constant *GlobalShadowAddress =
      F.getParent()->getOrInsertGlobal(kTysanShadowMemoryAddress, IntptrTy);
  return IRB.CreateLoad(IntptrTy, GlobalShadowAddress, "shadow.base");
}

Instruction *TypeSanitizer::getAppMemMask(Function &F) {
  IRBuilder<> IRB(&F.front().front());
  Value *GlobalAppMemMask =
      F.getParent()->getOrInsertGlobal(kTysanAppMemMask, IntptrTy);
  return IRB.CreateLoad(IntptrTy, GlobalAppMemMask, "app.mem.mask");
}

/// Collect all loads and stores, and for what TBAA nodes we need to generate
/// type descriptors.
void collectMemAccessInfo(
    Function &F, const TargetLibraryInfo &TLI,
    SmallVectorImpl<std::pair<Instruction *, MemoryLocation>> &MemoryAccesses,
    SmallSetVector<const MDNode *, 8> &TBAAMetadata,
    SmallVectorImpl<Value *> &MemTypeResetInsts) {
  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (Instruction &Inst : instructions(F)) {
    // Skip memory accesses inserted by another instrumentation.
    if (Inst.getMetadata(LLVMContext::MD_nosanitize))
      continue;

    if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst) ||
        isa<AtomicCmpXchgInst>(Inst) || isa<AtomicRMWInst>(Inst)) {
      MemoryLocation MLoc = MemoryLocation::get(&Inst);

      // Swift errors are special (we can't introduce extra uses on them).
      if (MLoc.Ptr->isSwiftError())
        continue;

      // Skip non-address-space-0 pointers; we don't know how to handle them.
      Type *PtrTy = cast<PointerType>(MLoc.Ptr->getType());
      if (PtrTy->getPointerAddressSpace() != 0)
        continue;

      if (MLoc.AATags.TBAA)
        TBAAMetadata.insert(MLoc.AATags.TBAA);
      MemoryAccesses.push_back(std::make_pair(&Inst, MLoc));
    } else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
      if (CallInst *CI = dyn_cast<CallInst>(&Inst))
        maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);

      if (isa<MemIntrinsic>(Inst)) {
        MemTypeResetInsts.push_back(&Inst);
      } else if (auto *II = dyn_cast<IntrinsicInst>(&Inst)) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end)
          MemTypeResetInsts.push_back(&Inst);
      }
    } else if (isa<AllocaInst>(Inst)) {
      MemTypeResetInsts.push_back(&Inst);
    }
  }
}

bool TypeSanitizer::run(Function &F, const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __tysan_init from within
  // the module constructor.
  if (&F == TysanCtorFunction.getCallee() || &F == TysanGlobalsSetTypeFunction)
    return false;
  initializeCallbacks(*F.getParent());

  // We need to collect all loads and stores, and know for what TBAA nodes we
  // need to generate type descriptors.
  SmallVector<std::pair<Instruction *, MemoryLocation>> MemoryAccesses;
  SmallSetVector<const MDNode *, 8> TBAAMetadata;
  SmallVector<Value *> MemTypeResetInsts;
  collectMemAccessInfo(F, TLI, MemoryAccesses, TBAAMetadata, MemTypeResetInsts);

  // byval arguments also need their types reset (they're new stack memory,
  // just like allocas).
  for (auto &A : F.args())
    if (A.hasByValAttr())
      MemTypeResetInsts.push_back(&A);

  Module &M = *F.getParent();
  TypeDescriptorsMapTy TypeDescriptors;
  TypeNameMapTy TypeNames;
  bool Res = false;
  for (const MDNode *MD : TBAAMetadata) {
    if (TypeDescriptors.count(MD))
      continue;

    if (!generateTypeDescriptor(MD, TypeDescriptors, TypeNames, M))
      return Res; // Giving up.

    Res = true;
  }

  const DataLayout &DL = F.getParent()->getDataLayout();
  bool SanitizeFunction = F.hasFnAttribute(Attribute::SanitizeType);
  bool NeedsInstrumentation =
      MemTypeResetInsts.empty() && MemoryAccesses.empty();
  Instruction *ShadowBase = NeedsInstrumentation ? nullptr : getShadowBase(F);
  Instruction *AppMemMask = NeedsInstrumentation ? nullptr : getAppMemMask(F);
  for (const auto &[I, MLoc] : MemoryAccesses) {
    IRBuilder<> IRB(I);
    assert(MLoc.Size.isPrecise());
    if (instrumentWithShadowUpdate(
            IRB, MLoc.AATags.TBAA, const_cast<Value *>(MLoc.Ptr),
            MLoc.Size.getValue(), I->mayReadFromMemory(), I->mayWriteToMemory(),
            ShadowBase, AppMemMask, false, SanitizeFunction, TypeDescriptors,
            DL)) {
      ++NumInstrumentedAccesses;
      Res = true;
    }
  }

  for (auto Inst : MemTypeResetInsts)
    Res |= instrumentMemInst(Inst, ShadowBase, AppMemMask, DL);

  return Res;
}

static Value *convertToShadowDataInt(IRBuilder<> &IRB, Value *Ptr,
                                     Type *IntptrTy, uint64_t PtrShift,
                                     Value *ShadowBase, Value *AppMemMask) {
  return IRB.CreateAdd(
      IRB.CreateShl(
          IRB.CreateAnd(IRB.CreatePtrToInt(Ptr, IntptrTy, "app.ptr.int"),
                        AppMemMask, "app.ptr.masked"),
          PtrShift, "app.ptr.shifted"),
      ShadowBase, "shadow.ptr.int");
}

bool TypeSanitizer::instrumentWithShadowUpdate(
    IRBuilder<> &IRB, const MDNode *TBAAMD, Value *Ptr, uint64_t AccessSize,
    bool IsRead, bool IsWrite, Value *ShadowBase, Value *AppMemMask,
    bool ForceSetType, bool SanitizeFunction,
    TypeDescriptorsMapTy &TypeDescriptors, const DataLayout &DL) {
  Constant *TDGV;
  if (TBAAMD)
    TDGV = TypeDescriptors[TBAAMD];
  else
    TDGV = Constant::getNullValue(IRB.getPtrTy());

  Value *TD = IRB.CreateBitCast(TDGV, IRB.getPtrTy());

  Value *ShadowDataInt = convertToShadowDataInt(IRB, Ptr, IntptrTy, PtrShift,
                                                ShadowBase, AppMemMask);
  Type *Int8PtrPtrTy = PointerType::get(IRB.getPtrTy(), 0);
  Value *ShadowData =
      IRB.CreateIntToPtr(ShadowDataInt, Int8PtrPtrTy, "shadow.ptr");

  auto SetType = [&]() {
    IRB.CreateStore(TD, ShadowData);

    // Now fill the remainder of the shadow memory corresponding to the
    // remainder of the the bytes of the type with a bad type descriptor.
    for (uint64_t i = 1; i < AccessSize; ++i) {
      Value *BadShadowData = IRB.CreateIntToPtr(
          IRB.CreateAdd(ShadowDataInt,
                        ConstantInt::get(IntptrTy, i << PtrShift),
                        "shadow.byte." + Twine(i) + ".offset"),
          Int8PtrPtrTy, "shadow.byte." + Twine(i) + ".ptr");

      // This is the TD value, -i, which is used to indicate that the byte is
      // i bytes after the first byte of the type.
      Value *BadTD =
          IRB.CreateIntToPtr(ConstantInt::getSigned(IntptrTy, -i),
                             IRB.getPtrTy(), "bad.descriptor" + Twine(i));
      IRB.CreateStore(BadTD, BadShadowData);
    }
  };

  if (ForceSetType || (ClWritesAlwaysSetType && IsWrite)) {
    // In the mode where writes always set the type, for a write (which does
    // not also read), we just set the type.
    SetType();
    return true;
  }

  assert((!ClWritesAlwaysSetType || IsRead) &&
         "should have handled case above");
  LLVMContext &C = IRB.getContext();
  MDNode *UnlikelyBW = MDBuilder(C).createBranchWeights(1, 100000);

  if (!SanitizeFunction) {
    // If we're not sanitizing this function, then we only care whether we
    // need to *set* the type.
    Value *LoadedTD = IRB.CreateLoad(IRB.getPtrTy(), ShadowData, "shadow.desc");
    Value *NullTDCmp = IRB.CreateIsNull(LoadedTD, "desc.set");
    Instruction *NullTDTerm = SplitBlockAndInsertIfThen(
        NullTDCmp, &*IRB.GetInsertPoint(), false, UnlikelyBW);
    IRB.SetInsertPoint(NullTDTerm);
    NullTDTerm->getParent()->setName("set.type");
    SetType();
    return true;
  }
  // We need to check the type here. If the type is unknown, then the read
  // sets the type. If the type is known, then it is checked. If the type
  // doesn't match, then we call the runtime (which may yet determine that
  // the mismatch is okay).
  //
  // The checks generated below have the following strucutre.
  //
  //   ; First we load the descriptor for the load from shadow memory and
  //   ; compare it against the type descriptor for the current access type.
  //   %shadow.desc = load ptr %shadow.data
  //   %bad.desc = icmp ne %shadow.desc, %td
  //   br %bad.desc, %bad.bb, %good.bb
  //
  // bad.bb:
  //   %shadow.desc.null = icmp eq %shadow.desc, null
  //   br %shadow.desc.null, %null.td.bb, %good.td.bb
  //
  // null.td.bb:
  //   ; The typ is unknown, set it if all bytes in the value are also unknown.
  //   ; To check, we load the shadow data for all bytes of the access. For the
  //   ; pseudo code below, assume an access of size 1.
  //   %shadow.data.int = add %shadow.data.int, 0
  //   %l = load (inttoptr %shadow.data.int)
  //   %is.not.null = icmp ne %l, null
  //   %not.all.unknown = %is.not.null
  //   br %no.all.unknown, before.set.type.bb
  //
  // before.set.type.bb:
  //   ; Call runtime to check mismatch.
  //   call void @__tysan_check()
  //   br %set.type.bb
  //
  // set.type.bb:
  //   ; Now fill the remainder of the shadow memory corresponding to the
  //   ; remainder of the the bytes of the type with a bad type descriptor.
  //   store %TD, %shadow.data
  //   br %continue.bb
  //
  // good.td.bb::
  //   ; We have a non-trivial mismatch. Call the runtime.
  //   call void @__tysan_check()
  //   br %continue.bb
  //
  // good.bb:
  //  ; We appear to have the right type. Make sure that all other bytes in
  //  ; the type are still marked as interior bytes. If not, call the runtime.
  //   %shadow.data.int = add %shadow.data.int, 0
  //   %l = load (inttoptr %shadow.data.int)
  //   %not.all.interior = icmp sge %l, 0
  //   br %not.all.interior, label %check.rt.bb, label %continue.bb
  //
  //  check.rt.bb:
  //   call void @__tysan_check()
  //   br %continue.bb

  Constant *Flags = ConstantInt::get(OrdTy, int(IsRead) | (int(IsWrite) << 1));

  Value *LoadedTD = IRB.CreateLoad(IRB.getPtrTy(), ShadowData, "shadow.desc");
  Value *BadTDCmp = IRB.CreateICmpNE(LoadedTD, TD, "bad.desc");
  Instruction *BadTDTerm, *GoodTDTerm;
  SplitBlockAndInsertIfThenElse(BadTDCmp, &*IRB.GetInsertPoint(), &BadTDTerm,
                                &GoodTDTerm, UnlikelyBW);
  IRB.SetInsertPoint(BadTDTerm);

  // We now know that the types did not match (we're on the slow path). If
  // the type is unknown, then set it.
  Value *NullTDCmp = IRB.CreateIsNull(LoadedTD);
  Instruction *NullTDTerm, *MismatchTerm;
  SplitBlockAndInsertIfThenElse(NullTDCmp, &*IRB.GetInsertPoint(), &NullTDTerm,
                                &MismatchTerm);

  // If the type is unknown, then set the type.
  IRB.SetInsertPoint(NullTDTerm);

  // We're about to set the type. Make sure that all bytes in the value are
  // also of unknown type.
  Value *Size = ConstantInt::get(OrdTy, AccessSize);
  Value *NotAllUnkTD = IRB.getFalse();
  for (uint64_t i = 1; i < AccessSize; ++i) {
    Value *UnkShadowData = IRB.CreateIntToPtr(
        IRB.CreateAdd(ShadowDataInt, ConstantInt::get(IntptrTy, i << PtrShift)),
        Int8PtrPtrTy);
    Value *ILdTD = IRB.CreateLoad(IRB.getPtrTy(), UnkShadowData);
    NotAllUnkTD = IRB.CreateOr(NotAllUnkTD, IRB.CreateIsNotNull(ILdTD));
  }

  Instruction *BeforeSetType = &*IRB.GetInsertPoint();
  Instruction *BadUTDTerm =
      SplitBlockAndInsertIfThen(NotAllUnkTD, BeforeSetType, false, UnlikelyBW);
  IRB.SetInsertPoint(BadUTDTerm);
  IRB.CreateCall(TysanCheck, {IRB.CreateBitCast(Ptr, IRB.getPtrTy()), Size,
                              (Value *)TD, (Value *)Flags});

  IRB.SetInsertPoint(BeforeSetType);
  SetType();

  // We have a non-trivial mismatch. Call the runtime.
  IRB.SetInsertPoint(MismatchTerm);
  IRB.CreateCall(TysanCheck, {IRB.CreateBitCast(Ptr, IRB.getPtrTy()), Size,
                              (Value *)TD, (Value *)Flags});

  // We appear to have the right type. Make sure that all other bytes in
  // the type are still marked as interior bytes. If not, call the runtime.
  IRB.SetInsertPoint(GoodTDTerm);
  Value *NotAllBadTD = IRB.getFalse();
  for (uint64_t i = 1; i < AccessSize; ++i) {
    Value *BadShadowData = IRB.CreateIntToPtr(
        IRB.CreateAdd(ShadowDataInt, ConstantInt::get(IntptrTy, i << PtrShift)),
        Int8PtrPtrTy);
    Value *ILdTD = IRB.CreatePtrToInt(
        IRB.CreateLoad(IRB.getPtrTy(), BadShadowData), IntptrTy);
    NotAllBadTD = IRB.CreateOr(
        NotAllBadTD, IRB.CreateICmpSGE(ILdTD, ConstantInt::get(IntptrTy, 0)));
  }

  Instruction *BadITDTerm = SplitBlockAndInsertIfThen(
      NotAllBadTD, &*IRB.GetInsertPoint(), false, UnlikelyBW);
  IRB.SetInsertPoint(BadITDTerm);
  IRB.CreateCall(TysanCheck, {IRB.CreateBitCast(Ptr, IRB.getPtrTy()), Size,
                              (Value *)TD, (Value *)Flags});
  return true;
}

bool TypeSanitizer::instrumentMemInst(Value *V, Instruction *ShadowBase,
                                      Instruction *AppMemMask,
                                      const DataLayout &DL) {
  BasicBlock::iterator IP;
  BasicBlock *BB;
  Function *F;

  if (auto *I = dyn_cast<Instruction>(V)) {
    IP = BasicBlock::iterator(I);
    BB = I->getParent();
    F = BB->getParent();
  } else {
    auto *A = cast<Argument>(V);
    F = A->getParent();
    BB = &F->getEntryBlock();
    IP = BB->getFirstInsertionPt();

    // Find the next insert point after both ShadowBase and AppMemMask.
    if (IP->comesBefore(ShadowBase))
      IP = ShadowBase->getNextNode()->getIterator();
    if (IP->comesBefore(AppMemMask))
      IP = AppMemMask->getNextNode()->getIterator();
  }

  Value *Dest, *Size, *Src = nullptr;
  bool NeedsMemMove = false;
  IRBuilder<> IRB(BB, IP);

  if (auto *A = dyn_cast<Argument>(V)) {
    assert(A->hasByValAttr() && "Type reset for non-byval argument?");

    Dest = A;
    Size =
        ConstantInt::get(IntptrTy, DL.getTypeAllocSize(A->getParamByValType()));
  } else {
    auto *I = cast<Instruction>(V);
    if (auto *MI = dyn_cast<MemIntrinsic>(I)) {
      if (MI->getDestAddressSpace() != 0)
        return false;

      Dest = MI->getDest();
      Size = MI->getLength();

      if (auto *MTI = dyn_cast<MemTransferInst>(MI)) {
        if (MTI->getSourceAddressSpace() == 0) {
          Src = MTI->getSource();
          NeedsMemMove = isa<MemMoveInst>(MTI);
        }
      }
    } else if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
          II->getIntrinsicID() != Intrinsic::lifetime_end)
        return false;

      Size = II->getArgOperand(0);
      Dest = II->getArgOperand(1);
    } else if (auto *AI = dyn_cast<AllocaInst>(I)) {
      // We need to clear the types for new stack allocations (or else we might
      // read stale type information from a previous function execution).

      IRB.SetInsertPoint(&*std::next(BasicBlock::iterator(I)));
      IRB.SetInstDebugLocation(I);

      Size = IRB.CreateMul(
          IRB.CreateZExtOrTrunc(AI->getArraySize(), IntptrTy),
          ConstantInt::get(IntptrTy,
                           DL.getTypeAllocSize(AI->getAllocatedType())));
      Dest = I;
    } else {
      return false;
    }
  }

  if (!ShadowBase)
    ShadowBase = getShadowBase(*F);
  if (!AppMemMask)
    AppMemMask = getAppMemMask(*F);

  Value *ShadowDataInt = IRB.CreateAdd(
      IRB.CreateShl(
          IRB.CreateAnd(IRB.CreatePtrToInt(Dest, IntptrTy), AppMemMask),
          PtrShift),
      ShadowBase);
  Value *ShadowData = IRB.CreateIntToPtr(ShadowDataInt, IRB.getPtrTy());

  if (!Src) {
    IRB.CreateMemSet(ShadowData, IRB.getInt8(0), IRB.CreateShl(Size, PtrShift),
                     Align(1ull << PtrShift));
    return true;
  }

  Value *SrcShadowDataInt = IRB.CreateAdd(
      IRB.CreateShl(
          IRB.CreateAnd(IRB.CreatePtrToInt(Src, IntptrTy), AppMemMask),
          PtrShift),
      ShadowBase);
  Value *SrcShadowData = IRB.CreateIntToPtr(SrcShadowDataInt, IRB.getPtrTy());

  if (NeedsMemMove) {
    IRB.CreateMemMove(ShadowData, Align(1ull << PtrShift), SrcShadowData,
                      Align(1ull << PtrShift), IRB.CreateShl(Size, PtrShift));
  } else {
    IRB.CreateMemCpy(ShadowData, Align(1ull << PtrShift), SrcShadowData,
                     Align(1ull << PtrShift), IRB.CreateShl(Size, PtrShift));
  }

  return true;
}

PreservedAnalyses TypeSanitizerPass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  TypeSanitizer TySan(*F.getParent());
  TySan.run(F, FAM.getResult<TargetLibraryAnalysis>(F));
  return PreservedAnalyses::none();
}

PreservedAnalyses ModuleTypeSanitizerPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  Function *TysanCtorFunction;
  std::tie(TysanCtorFunction, std::ignore) =
      createSanitizerCtorAndInitFunctions(M, kTysanModuleCtorName,
                                          kTysanInitName, /*InitArgTypes=*/{},
                                          /*InitArgs=*/{});

  TypeSanitizer TySan(M);
  TySan.instrumentGlobals(M);
  appendToGlobalCtors(M, TysanCtorFunction, 0);
  return PreservedAnalyses::none();
}
